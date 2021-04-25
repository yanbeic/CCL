from torchvision import get_image_backend

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5

from datasets.videodataset import AudioVideoDataset
from datasets.loader import VideoLoaderHDF5 as VideoLoaderHDF5_kinetics


def image_name_formatter(x):
    return f'image_{x:05d}.jpg'


def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'vggsound',
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb' and dataset_name != 'kinetics':
            loader = VideoLoaderHDF5()
        elif input_type == 'rgb' and dataset_name == 'kinetics':
            loader = VideoLoaderHDF5_kinetics()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.hdf5')

    training_data = VideoDataset(video_path,
                                 annotation_path,
                                 'training',
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform,
                                 video_loader=loader,
                                 video_path_formatter=video_path_formatter)
    return training_data


def get_training_av_data(video_path,
                         audio_path,
                         annotation_path,
                         dataset_name,
                         input_type,
                         file_type,
                         spatial_transform=None,
                         temporal_transform=None,
                         target_transform=None):

    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'vggsound'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb' and dataset_name != 'kinetics':
            loader = VideoLoaderHDF5()
        elif input_type == 'rgb' and dataset_name == 'kinetics':
            loader = VideoLoaderHDF5_kinetics()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.hdf5')
        audio_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.npy')

    training_data = AudioVideoDataset(video_path,
                                      audio_path,
                                      annotation_path,
                                      'training',
                                      spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform,
                                      target_transform=target_transform,
                                      video_loader=loader,
                                      video_path_formatter=video_path_formatter,
                                      audio_path_formatter=audio_path_formatter)
    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'vggsound'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb' and dataset_name != 'kinetics':
            loader = VideoLoaderHDF5()
        elif input_type == 'rgb' and dataset_name == 'kinetics':
            loader = VideoLoaderHDF5_kinetics()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.hdf5')

    validation_data = VideoDatasetMultiClips(video_path,
                                             annotation_path,
                                             'validation',
                                             spatial_transform=spatial_transform,
                                             temporal_transform=temporal_transform,
                                             target_transform=target_transform,
                                             video_loader=loader,
                                             video_path_formatter=video_path_formatter)
    return validation_data, collate_fn


def get_inference_data(video_path,
                       annotation_path,
                       dataset_name,
                       input_type,
                       file_type,
                       inference_subset,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 'vggsound'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5']
    assert inference_subset in ['train', 'val', 'test']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.hdf5')

    if inference_subset == 'train':
        subset = 'training'
    elif inference_subset == 'val':
        subset = 'validation'
    elif inference_subset == 'test':
        subset = 'testing'

    inference_data = VideoDatasetMultiClips(video_path,
                                            annotation_path,
                                            subset,
                                            spatial_transform=spatial_transform,
                                            temporal_transform=temporal_transform,
                                            target_transform=target_transform,
                                            video_loader=loader,
                                            video_path_formatter=video_path_formatter,
                                            target_type=['video_id', 'segment'])

    return inference_data, collate_fn
