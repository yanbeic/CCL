import json
import os
import torch
import torch.utils.data as data
from pathlib import Path
from .loader import VideoLoader
from .loader import AudioFeatureLoader
from random import randrange
import numpy as np


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))
    return video_ids, video_paths, annotations


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        self.data, self.class_names, self.n_videos = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader
        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class, n_videos

    def __loading(self, path, frame_indices):

        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target

    def __len__(self):
        return len(self.data)


def get_av_database(data, subset, root_path, audio_path, video_path_formatter, audio_path_formatter):
    video_ids = []
    video_paths = []
    audio_paths = []
    annotations = []

    ## filter the classes
    av_classes = sorted(audio_path.iterdir())
    av_classes = [x.name for x in av_classes]

    for key, value in data['database'].items():
        # remove classes without audio
        if value['annotations']['label'] in av_classes:
            this_subset = value['subset']
            if this_subset == subset:
                video_ids.append(key)
                annotations.append(value['annotations'])
                if 'video_path' in value:
                    video_paths.append(Path(value['video_path']))
                else:
                    label = value['annotations']['label']
                    video_paths.append(video_path_formatter(root_path, label, key))

                ### audio features
                audio_file = audio_path_formatter(audio_path, label, key)
                if os.path.isfile(audio_file):
                    audio_paths.append(audio_path_formatter(audio_path, label, key))
                else:
                    audio_paths.append(None)
    return video_ids, video_paths, audio_paths, annotations


class AudioVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 audio_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 audio_path_formatter=(lambda audio_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        self.data, self.class_names, self.n_videos = self.__make_av_dataset(
            root_path, audio_path, annotation_path, subset, video_path_formatter, audio_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader
        self.audio_loader = AudioFeatureLoader()
        self.target_type = target_type

    def __make_av_dataset(self, root_path, audio_path, annotation_path, subset,
                       video_path_formatter, audio_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, audio_paths, annotations = get_av_database(
            data, subset, root_path, audio_path, video_path_formatter, audio_path_formatter)

        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'audio': str(audio_paths[i]),
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)
        return dataset, idx_to_class, n_videos

    def __loading(self, path, frame_indices, audio_filename):

        clip = self.loader(path, frame_indices)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        audio = self.audio_loader(audio_filename)
        # audio feature should be 1X128 dimension
        # sampled one features from the all the temporal audio features
        if audio is None:
            audio_dim = 512
            audio = np.zeros(audio_dim, dtype=np.float32)  # AudioCNN14embed512

        if audio is not None and len(audio.shape) > 1:
            # audio = np.mean(audio, axis=0)
            ind = randrange(audio.shape[0])
            audio = audio[ind]
        return clip, audio

    def __getitem__(self, index):
        path = self.data[index]['video']
        audio_filename = self.data[index]['audio']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip, audio = self.__loading(path, frame_indices, audio_filename)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target, audio

    def __len__(self):
        return len(self.data)
