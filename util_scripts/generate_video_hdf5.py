import subprocess
import argparse
import h5py
import numpy as np
import os
import shutil
from pathlib import Path
from joblib import Parallel, delayed
from utils import get_n_frames, get_n_frames_hdf5


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)
    hdf5_path = dst_dir_path.parent / f'{dst_dir_path.name}.hdf5'

    try:
        ### check if the file exist
        n_frames = get_n_frames_hdf5(hdf5_path)
    except:
        if os.path.isfile(hdf5_path) is False:
            # if the file does not exist
            if ext != video_file_path.suffix:
                return
            dst_root_path / video_file_path
            ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                         '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                         'stream=width,height,avg_frame_rate,duration').split()
            ffprobe_cmd.append(str(video_file_path))

            p = subprocess.run(ffprobe_cmd, capture_output=True)
            res = p.stdout.decode('utf-8').splitlines()
            if len(res) < 4:
                return

            width = int(res[0])
            height = int(res[1])

            if width > height:
                vf_param = f'scale=-1:{size}'
            else:
                vf_param = f'scale={size}:-1'

            if fps > 0:
                vf_param += f',minterpolate={fps}'

            ################################################################################################################
            ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
            ffmpeg_cmd += ['-threads', '1', f'{dst_dir_path}/image_%05d.jpg']
            subprocess.run(ffmpeg_cmd)

            hdf5_path = dst_dir_path.parent / f'{dst_dir_path.name}.hdf5'
            try:
              with h5py.File(hdf5_path, 'w') as f:
                  dtype = h5py.special_dtype(vlen='uint8')
                  video = f.create_dataset('video',
                                           (len(list(dst_dir_path.glob('*.jpg'))),),
                                           dtype=dtype)
            except OSError as exc:
              if 'errno = 36' in exc.args[0]:
                  hdf5_path = dst_dir_path.parent / f'{dst_dir_path.name[:250]}.hdf5'
                  with h5py.File(hdf5_path, 'w') as f:
                      dtype = h5py.special_dtype(vlen='uint8')
                      video = f.create_dataset('video',
                                               (len(list(dst_dir_path.glob('*.jpg'))),),
                                               dtype=dtype)
              else:
                  raise

            for i, file_path in enumerate(sorted(dst_dir_path.glob('*.jpg'))):
              with file_path.open('rb') as f:
                  data = f.read()
              try:
                with h5py.File(hdf5_path, 'r+') as f:
                    video = f['video']
                    video[i] = np.frombuffer(data, dtype='uint8')
              except:
                print('could not write')

            else:
                shutil.rmtree(dst_dir_path)


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):

    if not class_dir_path.is_dir():
        return
    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    for video_file_path in sorted(class_dir_path.iterdir()):
        video_process(video_file_path, dst_class_path, ext, fps, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path',
                        default=None,
                        type=Path,
                        help='Directory path of videos')
    parser.add_argument('--dst_path',
                        default=None,
                        type=Path,
                        help='Directory path of jpg videos')
    parser.add_argument('--dataset',
                        default='',
                        type=str,
                        help='Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet)')
    parser.add_argument('--n_jobs',
                        default=1,
                        type=int,
                        help='Number of parallel jobs')
    parser.add_argument('--split',
                        default=0,
                        type=int,
                        help='Number of parallel jobs')
    parser.add_argument('--total_split',
                        default=1,
                        type=int,
                        help='Number of total jobs')
    parser.add_argument('--fps',
                        default=-1,
                        type=int,
                        help=('Frame rates of output videos. '
                              '-1 means original frame rates.'))
    parser.add_argument('--size',
                        default=240,
                        type=int,
                        help='Frame size of output videos.')
    args = parser.parse_args()

    if args.dataset in ['kinetics', 'mit', 'activitynet', 'vggsound']:
        ext = '.mp4'
    else:
        ext = '.avi'

    class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
    k = args.split
    interval = int(len(class_dir_paths) / args.total_split)
    if k + 1 == args.total_split:
        class_dir_paths = class_dir_paths[k * interval::]
    else:
        class_dir_paths = class_dir_paths[k * interval:(k + 1) * interval]

    status_list = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(class_process)(class_dir_path, args.dst_path, ext, args.fps, args.size)
        for class_dir_path in class_dir_paths)