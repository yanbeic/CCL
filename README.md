# Compositional Contrastive Learning

PyTorch implementation on [Distilling Audio-Visual Knowledge by Compositional Contrastive Learning](https://yanbeic.github.io/Doc/CVPR21-ChenY.pdf).

## Introduction 

Distilling knowledge from the pre-trained teacher models helps to learn a small student model that generalizes better. While existing works mostly focus on distilling knowledge within the same modality, we explore to distill the multi-modal knowledge available in video data (i.e. audio and vision). Specifically, we propose to transfer audio and visual knowledge from pre-trained image and audio teacher models to learn more expressive video representations.   

In multi-modal distillation, there often exists a semantic gap across modalities, e.g. a video shows 'applying lipstick' visually but carries the 'music' as audio. To ensure effective multi-modal distillation in the presence of a cross-modal semantic gap, we propose *compositional contrastive learning*, which features learnable compositional embeddings to close the cross-modal semantic gap, and a contrastive distillation objective to align different modalities jointly in the shared latent space.  

We demonstrate our method can distill knowledge from the audio and visual modalities to learn a stronger video model for recognition and retrieval tasks on action recognition video datasets. 

<p align="center">
<img src="https://github.com/yanbeic/CCL/blob/main/figure/overview.png" width="75%">
</p>

## Bibtex
Please cite our paper if you find it useful for your research.

```
@inproceedings{chen2021distilling,
  title={Distilling Audio-Visual Knowledge by Compositional Contrastive Learning},
  author={Chen, Yanbei and Xian, Yongqin and Koepke, Sophia and Shan, Ying and Akata, Zeynep},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021},
  organization={IEEE}
}
}
```

Codes will be released soon.