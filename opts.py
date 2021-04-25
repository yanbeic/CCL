import argparse
from pathlib import Path


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default=None,
                        type=Path,
                        help='Root directory path')
    parser.add_argument('--video_path',
                        default=None,
                        type=Path,
                        help='Directory path of videos')
    parser.add_argument('--audio_path',
                        default=None,
                        type=Path,
                        help='Directory path of audios')
    parser.add_argument('--annotation_path',
                        default=None,
                        type=Path,
                        help='Annotation file path')
    parser.add_argument('--result_path',
                        default=None,
                        type=Path,
                        help='Result directory path')
    parser.add_argument('--dataset',
                        default='kinetics',
                        type=str,
                        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--n_classes',
                        default=400,
                        type=int,
                        help=
                        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--n_pretrain_classes',
                        default=0,
                        type=int,
                        help=('Number of classes of pretraining task.'
                              'When using --pretrain_path, this must be set.'))
    parser.add_argument('--pretrain_path',
                        default=None,
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument('--ft_begin_module',
                        default='',
                        type=str,
                        help=('Module name of beginning of fine-tuning'
                              '(conv1, layer1, fc, denseblock1, classifier, ...).'
                              'The default means all layers are fine-tuned.'))
    parser.add_argument('--sample_size',
                        default=112,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--image_size',
                        default=112,
                        type=int,
                        help='Height and width of input images, '
                             'higher input size (e.g. 224) can boost performance')
    parser.add_argument('--sample_duration',
                        default=16,
                        type=int,
                        help='Temporal duration of inputs')
    parser.add_argument('--sample_t_stride',
                        default=1,
                        type=int,
                        help='If larger than 1, input frames are subsampled with the stride.')
    parser.add_argument('--train_crop',
                        default='random',
                        type=str,
                        help=('Spatial cropping method in training. '
                              'random is uniform. '
                              'corner is selection from 4 corners and 1 center. '
                              '(random | corner | center)'))
    parser.add_argument("--scale_h", type=int, default=128,
                        help="Scale image height to")
    parser.add_argument("--scale_w", type=int, default=171,
                        help="Scale image width to")
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')
    parser.add_argument('--no_hflip',
                        action='store_true',
                        help='If true holizontal flipping is not performed.')
    parser.add_argument('--colorjitter',
                        action='store_true',
                        help='If true colorjitter is performed.')
    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center)'))
    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--mean_dataset',
                        default='kinetics',
                        type=str,
                        help=('dataset for mean values of mean subtraction'
                              '(activitynet | kinetics | 0.5)'))
    parser.add_argument('--no_mean_norm',
                        action='store_true',
                        help='If true, inputs are not normalized by mean.')
    parser.add_argument('--no_std_norm',
                        action='store_true',
                        help='If true, inputs are not normalized by standard deviation.')
    parser.add_argument('--value_scale',
                        default=1,
                        type=int,
                        help=
                        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument('--multistep_milestones',
                        default=[150, 225], # [150, 250] # default
                        type=int,
                        nargs='+',
                        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--overwrite_milestones',
                        action='store_true',
                        help='If true, overwriting multistep_milestones when resuming training.')
    parser.add_argument('--plateau_patience',
                        default=10,
                        type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch Size')
    parser.add_argument('--inference_batch_size',
                        default=0,
                        type=int,
                        help='Batch Size for inference. 0 means this is the same as batch_size.')
    parser.add_argument('--batchnorm_sync',
                        action='store_true',
                        help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--n_epochs',
                        default=200,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    parser.add_argument('--resume_path',
                        default=None,
                        type=Path,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--inference',
                        action='store_true',
                        help='If true, inference is performed.')
    parser.add_argument('--inference_subset',
                        default='val',
                        type=str,
                        help='Used subset in inference (train | val | test)')
    parser.add_argument('--inference_stride',
                        default=16,
                        type=int,
                        help='Stride of sliding window in inference.')
    parser.add_argument('--inference_crop',
                        default='center',
                        type=str,
                        help=('Cropping method in inference. (center | nocrop)'
                              'When nocrop, fully convolutional inference is performed,'
                              'and mini-batch consists of clips of one video.'))
    parser.add_argument('--inference_no_average',
                        action='store_true',
                        help='If true, outputs for segments in a video are not averaged.')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=8,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--model',
                        default='r21d',
                        type=str,
                        help=
                        '(r21d)')
    parser.add_argument('--model_depth',
                        default=18,
                        type=int,
                        help='Depth of resnet (18 | 34)')
    parser.add_argument('--input_type',
                        default='rgb',
                        type=str,
                        help='(rgb | flow)')
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--accimage',
                        action='store_true',
                        help='If true, accimage is used to load images.')
    parser.add_argument('--output_topk',
                        default=5,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--file_type',
                        default='jpg',
                        type=str,
                        help='(jpg | hdf5)')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='If true, output tensorboard log file.')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--val_freq',
                        default=1,
                        type=int,
                        help='evaluate on validation set every x epochs')
    parser.add_argument('--loss_weight',
                        default=0.5,
                        type=float,
                        help='loss weight')
    parser.add_argument('--use_audio',
                        action='store_true',
                        help='If true, load audio features to compute additional regularization term.')
    parser.add_argument('--use_image',
                        action='store_true',
                        help='If true, load image features to compute additional regularization term.')
    parser.add_argument('--normalization',
                        action='store_false',
                        help='normalization on features')
    args = parser.parse_args()

    return args
