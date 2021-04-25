import torch
from torch import nn
from models import r21d
from models import composition_classifier


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1
    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    ft_begin_module = [ft_begin_module]
    if ft_begin_module != 'fc':
      ft_begin_module.append('fc')

    print(ft_begin_module)
    parameters = []
    add_flag = False
    for k, v in model.named_parameters():

        if get_module_name(k) in ft_begin_module:
            print(get_module_name(k))
            add_flag = True

        if add_flag:
            parameters.append({'params': v})
    return parameters


def generate_prediction(input_dim, num_classes, normalization=False):
  predictions = composition_classifier.generate_prediction(input_dim=input_dim, num_classes=num_classes, normalization=normalization)
  return predictions


def generate_model(opt, use_features=False, local_features=False):

    assert opt.model in [
        'r21d',
    ]

    model = r21d.generate_model(model_depth=opt.model_depth, num_classes=opt.n_classes,
                                use_features=use_features, local_features=local_features)
    return model


def load_pretrained_model(model, pretrain_path, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'])

        tmp_model = model
        tmp_model.fc = nn.Linear(tmp_model.fc.in_features, n_finetune_classes)
    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()
    return model
