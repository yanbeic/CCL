# Compositional Contrastive Learning

[PyTorch](https://pytorch.org/) implementation on [Distilling Audio-Visual Knowledge by Compositional Contrastive Learning](https://yanbeic.github.io/Doc/CVPR21-ChenY.pdf).

## Introduction 

Distilling knowledge from the pre-trained teacher models helps to learn a small student model that generalizes better. While existing works mostly focus on distilling knowledge within the same modality, we explore to distill the multi-modal knowledge available in video data (i.e. audio and vision). Specifically, we propose to transfer audio and visual knowledge from pre-trained image and audio teacher models to learn more expressive video representations.   

In multi-modal distillation, there often exists a semantic gap across modalities, e.g. a video shows *applying lipstick* visually while its accompanied audio is *music*. To ensure effective multi-modal distillation in the presence of a cross-modal semantic gap, we propose **compositional contrastive learning**, which features learnable compositional embeddings to close the cross-modal semantic gap, and a multi-class contrastive distillation objective to align different modalities jointly in the shared latent space.  

We demonstrate our method can distill knowledge from the audio and visual modalities to learn a stronger video model for recognition and retrieval tasks on video action recognition datasets. 

<p align="center">
<img src="https://github.com/yanbeic/CCL/blob/main/figure/overview.png" width="75%">
</p>

## Getting Started
### Prerequisites:
- python >= 3.6.10 
- pytorch >= 1.1.0
- FFmpeg, FFprobe
- Download datasets: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler), [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)

### Data Preparation on UCF101 (example):
- audio features are extracted based on the audio pre-trained model [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn). The UCF101 audio features are provided under the directory `dataset/UCF101`. Please uncompress the `audiocnn14embed512_features.tar.gz` file for details. 
- video data is convert to the `hdf5` format using the following command. Please specify the data directory `${UCF101_DATA_DIR}`, e.g. `datasets/UCF101/UCF-101`. Note: video data can be downloaded [here](https://www.crcv.ucf.edu/data/UCF101.php).
```
python util_scripts/generate_video_hdf5.py --dir_path=${UCF101_DATA_DIR} --dst_path=datasets/UCF101/hdf5data --dataset=ucf101
```
- prepare the `json` file for dataloader using the following command. Note: official data splits can be downloaded [here](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip).
```
python util_scripts/ucf101_json.py --dir_path=datasets/UCF101/ucfTrainTestlist --video_path=datasets/UCF101/hdf5data --audio_path=datasets/UCF101/audiocnn14embed512_features --dst_path=datasets/UCF101/ --video_type=hdf5
```

### Training & Testing:

The running commands for both training and testing are written in the same script file. Experiments are conducted on 2 gpus. Please refer to the script files in the directory `scripts` for details. Use the folllowing commands to test on the UCF51 dataset.

- baseline (w/o distillation)
```
sh scripts/run_baseline.sh
```

- CCL (A): distilling audio knowledge from the pre-trained audio teacher model (audiocnn14)
```
sh scripts/run_ccl_audio.sh
```

- CCL (I): distilling image knowledge from the pre-trained image teacher model (resnet34)
```
sh scripts/run_ccl_image.sh
```

- CCL (AI): distilling audio and image knowledge from the pre-trained audio and image teacher models
```
sh scripts/run_ccl_ai.sh
```


## Bibtex
```
@inproceedings{chen2021distilling,
  title={Distilling Audio-Visual Knowledge by Compositional Contrastive Learning},
  author={Chen, Yanbei and Xian, Yongqin and Koepke, Sophia and Shan, Ying and Akata, Zeynep},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgement
This repository is partially built with two open-source implementation: (1) [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) is used in video data preparation; (2) [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) is used for audio feature extraction.