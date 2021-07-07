import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import random
from utils import AverageMeter, calculate_accuracy, write_to_batch_logger, write_to_epoch_logger


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False):
    print('train at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        outputs, features = model(inputs)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.
              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss=losses,
                     acc=accuracies), flush=True)

    if distributed:
        loss_sum = torch.tensor([losses.sum], dtype=torch.float32, device=device)
        loss_count = torch.tensor([losses.count], dtype=torch.float32, device=device)
        acc_sum = torch.tensor([accuracies.sum], dtype=torch.float32, device=device)
        acc_count = torch.tensor([accuracies.count], dtype=torch.float32, device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    write_to_epoch_logger(epoch_logger, epoch, losses.val, accuracies.val, current_lr)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)


def train_a_epoch(epoch,
                  data_loader,
                  model,
                  joint_prediction_aud,
                  criterion,
                  criterion_jsd,
                  criterion_ct_av,
                  optimizer,
                  optimizer_av,
                  device,
                  current_lr,
                  epoch_logger,
                  batch_logger,
                  tb_writer=None,
                  distributed=False):
    print('train at epoch {}'.format(epoch))
    model.train()
    joint_prediction_aud.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # classification loss
    losses_cls = AverageMeter()
    accuracies = AverageMeter()
    # contrastive loss
    losses_ct_av = AverageMeter()
    # jsd loss
    losses_jsd_a = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, audios) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        outputs, features = model(inputs)
        targets = targets.to(device, non_blocking=True)
        audios = audios.to(device, non_blocking=True)

        loss_cls_v = criterion(outputs, targets) # video classification loss
        acc = calculate_accuracy(outputs, targets)
        #####################################################################################
        # use audio features as features & filter out the zero-ones (not available) audio features
        features_aud = audios[audios.sum(dim=1) != 0]
        features_vid = features[audios.sum(dim=1) != 0]
        targets_new = targets[audios.sum(dim=1) != 0]
        outputs_new = outputs[audios.sum(dim=1) != 0]

        # here compose images and videos
        outputs_av, features_av = joint_prediction_aud(features_aud, features_vid)
        loss_cls_a = criterion(outputs_av, targets_new) # video classification loss

        # contrastive learning (symmetric loss)
        # align video features to multimodal (audio-video) features
        loss_vm = criterion_ct_av(features_vid, features_av, targets_new) + criterion_ct_av(features_av, features_vid, targets_new)
        # align video features to audio features
        loss_va = criterion_ct_av(features_vid, features_aud, targets_new) + criterion_ct_av(features_aud, features_vid, targets_new)
        # align multimodal (audio-video) features to audio features
        loss_ma = criterion_ct_av(features_av, features_aud, targets_new) + criterion_ct_av(features_aud, features_av, targets_new)
        # contrastive loss
        loss_ct_av = loss_vm + loss_va
        loss_ct_a = loss_vm + loss_ma
        # jsd loss
        loss_jsd_a = criterion_jsd(outputs_new, outputs_av)
        #####################################################################################
        total_loss_v = sum([loss_cls_v, loss_ct_av, loss_jsd_a])
        total_loss_a = sum([loss_cls_a, loss_ct_a, loss_jsd_a])

        losses_cls.update(loss_cls_v.item(), inputs.size(0))
        losses_ct_av.update(loss_ct_av.item(), inputs.size(0))
        losses_jsd_a.update(loss_jsd_a.item(), inputs.size(0))

        accuracies.update(acc, inputs.size(0))

        optimizer_av.zero_grad()
        total_loss_a.backward(retain_graph=True)

        optimizer.zero_grad()
        total_loss_v.backward()

        optimizer_av.step()
        optimizer.step()
        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_cls {loss_cls.val:.3f} ({loss_cls.avg:.3f})\t'
              'Loss_ct_a {loss_ct_av.val:.3f} ({loss_ct_av.avg:.3f})\t'
              'Loss_jsd_a {loss_jsd_a.val:.3f} ({loss_jsd_a.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.
              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss_cls=losses_cls,
                     loss_ct_av=losses_ct_av,
                     loss_jsd_a=losses_jsd_a,
                     acc=accuracies), flush=True)

        if distributed:
            loss_cls_sum = torch.tensor([losses_cls.sum], dtype=torch.float32, device=device)
            loss_ct_av_sum = torch.tensor([losses_ct_av.sum], dtype=torch.float32, device=device)
            loss_jsd_a_sum = torch.tensor([losses_jsd_a.sum], dtype=torch.float32, device=device)
            acc_sum = torch.tensor([accuracies.sum], dtype=torch.float32, device=device)
            loss_count = torch.tensor([losses_cls.count], dtype=torch.float32, device=device)
            acc_count = torch.tensor([accuracies.count], dtype=torch.float32, device=device)

            dist.all_reduce(loss_cls_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_ct_av_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_jsd_a_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

            losses_cls.avg = loss_cls_sum.item() / loss_count.item()
            losses_ct_av.avg = loss_ct_av_sum.item() / loss_count.item()
            losses_jsd_a.avg = loss_jsd_a_sum.item() / loss_count.item()
            accuracies.avg = acc_sum.item() / acc_count.item()

        write_to_epoch_logger(epoch_logger, epoch, losses_cls.val, accuracies.val, current_lr)

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_cls', losses_cls.avg, epoch)
            tb_writer.add_scalar('train/loss_ct_av', losses_ct_av.avg, epoch)
            tb_writer.add_scalar('train/loss_jsd_a', losses_jsd_a.avg, epoch)
            tb_writer.add_scalar('train/acc', accuracies.avg, epoch)


def train_i_epoch(epoch,
                  data_loader,
                  model,
                  image_model,
                  joint_prediction_img,
                  criterion,
                  criterion_jsd,
                  criterion_ct_iv,
                  optimizer,
                  optimizer_iv,
                  device,
                  current_lr,
                  epoch_logger,
                  batch_logger,
                  tb_writer=None,
                  distributed=False,
                  image_size=None):
    print('train at epoch {}'.format(epoch))
    model.train()
    image_model.eval()
    joint_prediction_img.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # classification loss
    losses_cls = AverageMeter()
    accuracies = AverageMeter()
    # contrastive loss
    losses_ct_iv = AverageMeter()
    # jsd loss
    losses_jsd_i = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        outputs, features = model(inputs)
        targets = targets.to(device, non_blocking=True)

        loss_cls_v = criterion(outputs, targets) # video classification loss
        acc = calculate_accuracy(outputs, targets)
        #####################################################################################
        if image_size is not None:
            # here resize the image to a larger size than the video input size (better fit the resnet)
            inputs = F.interpolate(inputs, size=[inputs.shape[0], image_size, image_size])
        ### randomly select an image
        rand_img = random.randint(0, inputs.shape[2] - 1)
        images = inputs[:, :, rand_img, :, :]
        features_img = image_model(images)
        features_img = features_img.squeeze()

        # here compose images and videos
        outputs_iv, features_iv = joint_prediction_img(features_img, features)
        loss_cls_i = criterion(outputs_iv, targets) # video classification loss

        # contrastive learning (symmetric loss)
        # align video features to multimodal (image-video) features
        loss_vm = criterion_ct_iv(features, features_iv, targets) + criterion_ct_iv(features_iv, features, targets)
        # align video features to image features
        loss_vi = criterion_ct_iv(features, features_img, targets) + criterion_ct_iv(features_img, features, targets)
        # align multimodal features to image features
        loss_mi = criterion_ct_iv(features_iv, features_img, targets) + criterion_ct_iv(features_img, features_iv, targets)
        # contrastive loss
        loss_ct_iv = loss_vm + loss_vi
        loss_ct_i = loss_vm + loss_mi
        # jsd loss
        loss_jsd_i = criterion_jsd(outputs, outputs_iv)
        #####################################################################################
        total_loss_v = sum([loss_cls_v, loss_ct_iv, loss_jsd_i])
        total_loss_i = sum([loss_cls_i, loss_ct_i, loss_jsd_i])

        losses_cls.update(loss_cls_v.item(), inputs.size(0))
        losses_ct_iv.update(loss_ct_iv.item(), inputs.size(0))
        losses_jsd_i.update(loss_jsd_i.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer_iv.zero_grad()
        total_loss_i.backward(retain_graph=True)

        optimizer.zero_grad()
        total_loss_v.backward()

        optimizer_iv.step()
        optimizer.step()
        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_cls {loss_cls.val:.3f} ({loss_cls.avg:.3f})\t'
              'Loss_ct_i {loss_ct_iv.val:.3f} ({loss_ct_iv.avg:.3f})\t'
              'Loss_jsd_i {loss_jsd_i.val:.3f} ({loss_jsd_i.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.
              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss_cls=losses_cls,
                     loss_ct_iv=losses_ct_iv,
                     loss_jsd_i=losses_jsd_i,
                     acc=accuracies), flush=True)

        if distributed:
            loss_cls_sum = torch.tensor([losses_cls.sum], dtype=torch.float32, device=device)
            loss_ct_iv_sum = torch.tensor([losses_ct_iv.sum], dtype=torch.float32, device=device)
            loss_jsd_i_sum = torch.tensor([losses_jsd_i.sum], dtype=torch.float32, device=device)
            acc_sum = torch.tensor([accuracies.sum], dtype=torch.float32, device=device)
            loss_count = torch.tensor([losses_cls.count], dtype=torch.float32, device=device)
            acc_count = torch.tensor([accuracies.count], dtype=torch.float32, device=device)

            dist.all_reduce(loss_cls_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_ct_iv_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_jsd_i_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

            losses_cls.avg = loss_cls_sum.item() / loss_count.item()
            losses_ct_iv.avg = loss_ct_iv_sum.item() / loss_count.item()
            losses_jsd_i.avg = loss_jsd_i_sum.item() / loss_count.item()
            accuracies.avg = acc_sum.item() / acc_count.item()

        write_to_epoch_logger(epoch_logger, epoch, losses_cls.val, accuracies.val, current_lr)

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_cls', losses_cls.avg, epoch)
            tb_writer.add_scalar('train/loss_ct_iv', losses_ct_iv.avg, epoch)
            tb_writer.add_scalar('train/loss_jsd_i', losses_jsd_i.avg, epoch)
            tb_writer.add_scalar('train/acc', accuracies.avg, epoch)


def train_ai_epoch(epoch,
                   data_loader,
                   model,
                   image_model,
                   joint_prediction_aud,
                   joint_prediction_img,
                   criterion,
                   criterion_jsd,
                   criterion_ct_av,
                   criterion_ct_iv,
                   optimizer,
                   optimizer_av,
                   optimizer_iv,
                   device,
                   current_lr,
                   epoch_logger,
                   batch_logger,
                   tb_writer=None,
                   distributed=False,
                   image_size=None,
                   loss_weight=1.0):
    print('train at epoch {}'.format(epoch))
    model.train()
    image_model.eval()
    joint_prediction_aud.train()
    joint_prediction_img.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # classification loss
    losses_cls = AverageMeter()
    accuracies = AverageMeter()
    # contrastive loss
    losses_ct_av = AverageMeter()
    losses_ct_iv = AverageMeter()
    # jsd loss
    losses_jsd_a = AverageMeter()
    losses_jsd_i = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, audios) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        outputs, features = model(inputs)
        targets = targets.to(device, non_blocking=True)
        audios = audios.to(device, non_blocking=True)

        loss_cls_v = criterion(outputs, targets) # video classification loss
        acc = calculate_accuracy(outputs, targets)
        #####################################################################################
        if image_size is not None:
            # here resize the image to a larger size than the video input size (better fit the resnet)
            inputs = F.interpolate(inputs, size=[inputs.shape[0], image_size, image_size])
        ### randomly select an image
        rand_img = random.randint(0, inputs.shape[2] - 1)
        images = inputs[:, :, rand_img, :, :]
        features_img = image_model(images)
        features_img = features_img.squeeze()

        # here compose images and videos
        outputs_iv, features_iv = joint_prediction_img(features_img, features)
        loss_cls_i = criterion(outputs_iv, targets) # video classification loss

        # contrastive learning (symmetric loss)
        # align video features to multimodal (image-video) features
        loss_vm = criterion_ct_iv(features, features_iv, targets) + criterion_ct_iv(features_iv, features, targets)
        # align video features to image features
        loss_vi = criterion_ct_iv(features, features_img, targets) + criterion_ct_iv(features_img, features, targets)
        # align multimodal features to image features
        loss_mi = criterion_ct_iv(features_iv, features_img, targets) + criterion_ct_iv(features_img, features_iv, targets)
        # contrastive loss
        loss_ct_iv = loss_vm + loss_vi
        loss_ct_i = loss_vm + loss_mi
        # jsd loss
        loss_jsd_i = criterion_jsd(outputs, outputs_iv)
        #####################################################################################
        # use audio features as features & filter out the zero-ones (not available) audio features
        features_aud = audios[audios.sum(dim=1) != 0]
        features_vid = features[audios.sum(dim=1) != 0]
        targets_new = targets[audios.sum(dim=1) != 0]
        outputs_new = outputs[audios.sum(dim=1) != 0]

        # here compose images and videos
        outputs_av, features_av = joint_prediction_aud(features_aud, features_vid)
        loss_cls_a = criterion(outputs_av, targets_new) # video classification loss

        # contrastive learning (symmetric loss)
        # align video features to multimodal (audio-video) features
        loss_vm = criterion_ct_av(features_vid, features_av, targets_new) + criterion_ct_av(features_av, features_vid, targets_new)
        # align video features to audio features
        loss_va = criterion_ct_av(features_vid, features_aud, targets_new) + criterion_ct_av(features_aud, features_vid, targets_new)
        # align multimodal (audio-video) features to audio features
        loss_ma = criterion_ct_av(features_av, features_aud, targets_new) + criterion_ct_av(features_aud, features_av, targets_new)
        # contrastive loss
        loss_ct_av = loss_vm + loss_va
        loss_ct_a = loss_vm + loss_ma
        # jsd loss
        loss_jsd_a = criterion_jsd(outputs_new, outputs_av)
        #####################################################################################
        total_loss_v = sum([loss_cls_v, (loss_ct_av + loss_ct_iv) * loss_weight, (loss_jsd_a + loss_jsd_i) * loss_weight])
        total_loss_a = sum([loss_cls_a, loss_ct_a, loss_jsd_a])
        total_loss_i = sum([loss_cls_i, loss_ct_i, loss_jsd_i])

        losses_cls.update(loss_cls_v.item(), inputs.size(0))
        losses_ct_av.update(loss_ct_av.item(), inputs.size(0))
        losses_ct_iv.update(loss_ct_iv.item(), inputs.size(0))
        losses_jsd_a.update(loss_jsd_a.item(), inputs.size(0))
        losses_jsd_i.update(loss_jsd_i.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer_av.zero_grad()
        total_loss_a.backward(retain_graph=True)

        optimizer_iv.zero_grad()
        total_loss_i.backward(retain_graph=True)

        optimizer.zero_grad()
        total_loss_v.backward()

        optimizer_av.step()
        optimizer_iv.step()
        optimizer.step()
        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_cls {loss_cls.val:.3f} ({loss_cls.avg:.3f})\t'
              'Loss_ct_a {loss_ct_av.val:.3f} ({loss_ct_av.avg:.3f})\t'
              'Loss_ct_i {loss_ct_iv.val:.3f} ({loss_ct_iv.avg:.3f})\t'
              'Loss_jsd_a {loss_jsd_a.val:.3f} ({loss_jsd_a.avg:.3f})\t'
              'Loss_jsd_i {loss_jsd_i.val:.3f} ({loss_jsd_i.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.
              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss_cls=losses_cls,
                     loss_ct_av=losses_ct_av,
                     loss_ct_iv=losses_ct_iv,
                     loss_jsd_a=losses_jsd_a,
                     loss_jsd_i=losses_jsd_i,
                     acc=accuracies), flush=True)

        if distributed:
            loss_cls_sum = torch.tensor([losses_cls.sum], dtype=torch.float32, device=device)
            loss_ct_av_sum = torch.tensor([losses_ct_av.sum], dtype=torch.float32, device=device)
            loss_ct_iv_sum = torch.tensor([losses_ct_iv.sum], dtype=torch.float32, device=device)
            loss_jsd_a_sum = torch.tensor([losses_jsd_a.sum], dtype=torch.float32, device=device)
            loss_jsd_i_sum = torch.tensor([losses_jsd_i.sum], dtype=torch.float32, device=device)
            acc_sum = torch.tensor([accuracies.sum], dtype=torch.float32, device=device)
            loss_count = torch.tensor([losses_cls.count], dtype=torch.float32, device=device)
            acc_count = torch.tensor([accuracies.count], dtype=torch.float32, device=device)

            dist.all_reduce(loss_cls_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_ct_av_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_ct_iv_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_jsd_a_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_jsd_i_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

            losses_cls.avg = loss_cls_sum.item() / loss_count.item()
            losses_ct_av.avg = loss_ct_av_sum.item() / loss_count.item()
            losses_ct_iv.avg = loss_ct_iv_sum.item() / loss_count.item()
            losses_jsd_a.avg = loss_jsd_a_sum.item() / loss_count.item()
            losses_jsd_i.avg = loss_jsd_i_sum.item() / loss_count.item()
            accuracies.avg = acc_sum.item() / acc_count.item()

        write_to_epoch_logger(epoch_logger, epoch, losses_cls.val, accuracies.val, current_lr)

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_cls', losses_cls.avg, epoch)
            tb_writer.add_scalar('train/loss_ct_av', losses_ct_av.avg, epoch)
            tb_writer.add_scalar('train/loss_ct_iv', losses_ct_iv.avg, epoch)
            tb_writer.add_scalar('train/loss_jsd_a', losses_jsd_a.avg, epoch)
            tb_writer.add_scalar('train/loss_jsd_i', losses_jsd_i.avg, epoch)
            tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
