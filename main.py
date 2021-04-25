import json
import random
import os
import numpy as np
import torch
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from pathlib import Path

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel, generate_prediction,
                   get_fine_tuning_parameters)
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop, RandomCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from dataset import get_training_av_data
from training import train_epoch, train_a_epoch, train_i_epoch, train_ai_epoch
from validation import val_epoch
from loss.jsd_loss import JSDLoss
from loss.nce_loss import NCELoss
import inference


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        if opt.audio_path is not None:
            opt.audio_path = opt.root_path / opt.audio_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)
    return opt


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters, av_parameters, iv_parameters):
    assert opt.train_crop in ['random', 'corner', 'center', 'other']
    spatial_transform = []
    if opt.train_crop == 'random':
        print('random crop')
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    elif opt.train_crop == 'other':
        print('other')
        spatial_transform.append(Resize((opt.scale_h, opt.scale_w)))
        spatial_transform.append(RandomCrop(opt.sample_size))

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    # whether additional audio features are used
    if av_parameters is None:
        train_data = get_training_data(opt.video_path, opt.annotation_path,
                                       opt.dataset, opt.input_type, opt.file_type,
                                       spatial_transform, temporal_transform)
    else:
        train_data = get_training_av_data(opt.video_path, opt.audio_path, opt.annotation_path,
                                          opt.dataset, opt.input_type, opt.file_type,
                                          spatial_transform, temporal_transform)

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    if opt.use_audio:
        optimizer_av = SGD(av_parameters,
                           lr=opt.learning_rate,
                           momentum=opt.momentum,
                           dampening=dampening,
                           weight_decay=opt.weight_decay,
                           nesterov=opt.nesterov)
    else:
        optimizer_av = None

    if opt.use_image:
        optimizer_iv = SGD(iv_parameters,
                           lr=opt.learning_rate,
                           momentum=opt.momentum,
                           dampening=dampening,
                           weight_decay=opt.weight_decay,
                           nesterov=opt.nesterov)
    else:
        optimizer_iv = None

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)
    else:
        print(opt.multistep_milestones)
        scheduler = lr_scheduler.MultiStepLR(optimizer, opt.multistep_milestones)
    return (train_loader, train_sampler, train_logger, train_batch_logger, optimizer, optimizer_av, optimizer_iv, scheduler)


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    if opt.train_crop == 'other':
        spatial_transform = [
            Resize((opt.scale_h, opt.scale_w)),
            RandomCrop(opt.sample_size),
            ToTensor()
        ]
    else:
        spatial_transform = [
            Resize(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor()
        ]

    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))

    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))

    temporal_transform = TemporalCompose(temporal_transform)

    val_data, collate_fn = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               spatial_transform, temporal_transform)

    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size // opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc', 'acc_num'])
    else:
        val_logger = None
    return val_loader, val_logger


def get_inference_utils(opt):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    if opt.train_crop == 'other':
        spatial_transform = [
            Resize((opt.scale_h, opt.scale_w)),
            RandomCrop(opt.sample_size),
            ToTensor()
        ]
    else:
        spatial_transform = [Resize(opt.sample_size)]

    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    inference_data, collate_fn = get_inference_data(opt.video_path,
                                                    opt.annotation_path,
                                                    opt.dataset, opt.input_type,
                                                    opt.file_type, opt.inference_subset,
                                                    spatial_transform, temporal_transform)

    inference_loader = torch.utils.data.DataLoader(inference_data,
                                                   batch_size=opt.inference_batch_size,
                                                   shuffle=False,
                                                   num_workers=opt.n_threads,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn,
                                                   collate_fn=collate_fn)
    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    if opt.inference:
        model = generate_model(opt)
    else:
        model = generate_model(opt, use_features=True)

    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)

    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    else:
        parameters = model.parameters()

    if opt.is_master_node:
        print(model)

    #####################################################################################
    ### here add a classifier to predict videos and audios
    if opt.inference is False:
        ### define loss
        criterion = CrossEntropyLoss().to(opt.device)

        if opt.use_audio or opt.use_image:
            criterion_jsd = JSDLoss(weight=0.5)

        #################################################################################
        if opt.use_audio:
            ### define loss
            criterion_ct_av = NCELoss(temperature=0.5)
            ### audio teacher model
            feature_dim = 512 * 2
            if opt.pretrain_path is not None:
                joint_prediction_aud = generate_prediction(feature_dim, opt.n_finetune_classes, normalization=True)
            else:
                joint_prediction_aud = generate_prediction(feature_dim, opt.n_classes, normalization=True)
            if opt.resume_path is not None:
                aux_checkpoint = Path(os.path.join(str(opt.resume_path.parent), str(opt.resume_path.name[:-4] + '_audio.pth')))
                joint_prediction_aud = resume_model(aux_checkpoint, opt.arch, joint_prediction_aud)

            joint_prediction_aud = make_data_parallel(joint_prediction_aud, opt.distributed, opt.device)
            aud_para = joint_prediction_aud.parameters()
            joint_prediction_aud.cuda()
        else:
            aud_para = None

        #################################################################################
        if opt.use_image:
            ### define loss
            criterion_ct_iv = NCELoss(temperature=0.1)
            ### image teacher model
            image_model = torchvision.models.resnet34(pretrained=True)
            # remove the fc layers (only use the image features)
            image_model = torch.nn.Sequential(*list(image_model.children())[:-1])
            image_model = make_data_parallel(image_model, opt.distributed, opt.device)
            feature_dim = 512 * 2
            if opt.pretrain_path is not None:
                joint_prediction_img = generate_prediction(feature_dim, opt.n_finetune_classes, normalization=True)
            else:
                joint_prediction_img = generate_prediction(feature_dim, opt.n_classes, normalization=True)
            if opt.resume_path is not None:
                aux_checkpoint = Path(os.path.join(str(opt.resume_path.parent), str(opt.resume_path.name[:-4] + '_image.pth')))
                joint_prediction_img = resume_model(aux_checkpoint, opt.arch, joint_prediction_img)

            joint_prediction_img = make_data_parallel(joint_prediction_img, opt.distributed, opt.device)
            img_para = joint_prediction_img.parameters()
            joint_prediction_img.cuda()
        else:
            img_para = None

        #################################################################################
        (train_loader, train_sampler, train_logger, train_batch_logger, optimizer, optimizer_av, optimizer_iv, scheduler) = \
            get_train_utils(opt, model_parameters=parameters, av_parameters=aud_para, iv_parameters=img_para)

        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones

    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None
    pre_val_acc = 0.0
    if opt.image_size > opt.sample_size:
        image_size = opt.image_size
    else:
        image_size = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
            current_lr = get_lr(optimizer)
            if optimizer_av is None and optimizer_iv is None:
                train_epoch(epoch=i, data_loader=train_loader, model=model,
                            criterion=criterion, optimizer=optimizer,
                            device=opt.device, current_lr=current_lr,
                            epoch_logger=train_logger, batch_logger=train_batch_logger,
                            tb_writer=tb_writer, distributed=opt.distributed)
            elif optimizer_av is not None and optimizer_iv is None:
                train_a_epoch(epoch=i, data_loader=train_loader, model=model, joint_prediction_aud=joint_prediction_aud,
                              criterion=criterion, criterion_jsd=criterion_jsd, criterion_ct_av=criterion_ct_av,
                              optimizer=optimizer, optimizer_av=optimizer_av,
                              device=opt.device, current_lr=current_lr,
                              epoch_logger=train_logger, batch_logger=train_batch_logger,
                              tb_writer=tb_writer, distributed=opt.distributed)
            elif optimizer_av is None and optimizer_iv is not None:
                train_i_epoch(epoch=i, data_loader=train_loader, model=model, image_model=image_model, joint_prediction_img=joint_prediction_img,
                              criterion=criterion, criterion_jsd=criterion_jsd, criterion_ct_iv=criterion_ct_iv,
                              optimizer=optimizer, optimizer_iv=optimizer_iv,
                              device=opt.device, current_lr=current_lr,
                              epoch_logger=train_logger, batch_logger=train_batch_logger,
                              tb_writer=tb_writer, distributed=opt.distributed, image_size=image_size)
            else:
                train_ai_epoch(epoch=i, data_loader=train_loader, model=model, image_model=image_model,
                               joint_prediction_aud=joint_prediction_aud, joint_prediction_img=joint_prediction_img,
                               criterion=criterion, criterion_jsd=criterion_jsd, criterion_ct_av=criterion_ct_av, criterion_ct_iv=criterion_ct_iv,
                               optimizer=optimizer, optimizer_av=optimizer_av, optimizer_iv=optimizer_iv,
                               device=opt.device, current_lr=current_lr,
                               epoch_logger=train_logger, batch_logger=train_batch_logger,
                               tb_writer=tb_writer, distributed=opt.distributed, image_size=image_size, loss_weight=opt.loss_weight)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer, scheduler)
                if opt.use_audio:
                    save_file_path = opt.result_path / 'save_{}_audio.pth'.format(i)
                    save_checkpoint(save_file_path, i, opt.arch, joint_prediction_aud, optimizer, scheduler)
                if opt.use_image:
                    save_file_path = opt.result_path / 'save_{}_image.pth'.format(i)
                    save_checkpoint(save_file_path, i, opt.arch, joint_prediction_img, optimizer, scheduler)
            if not opt.no_val and i % opt.val_freq == 0:
                prev_val_loss, val_acc = val_epoch(i, val_loader, model, criterion, opt.device, val_logger, tb_writer, opt.distributed)
                if pre_val_acc < val_acc:
                    pre_val_acc = val_acc
                    save_file_path = opt.result_path / 'save_model.pth'
                    save_checkpoint(save_file_path, i, opt.arch, model, optimizer, scheduler)

            if not opt.no_train and opt.lr_scheduler == 'multistep':
                scheduler.step()
            elif not opt.no_train and opt.lr_scheduler == 'plateau':
                if prev_val_loss is not None:
                    scheduler.step(prev_val_loss)

    if opt.inference:
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(opt.inference_subset)
        inference.inference(inference_loader, model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk)


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')

    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)
