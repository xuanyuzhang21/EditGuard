<div align="center">
<img src="./asserts/Logo.png" alt="Image Alt Text" width="150" height="150">
<h3> EditGuard: Versatile Image Watermarking for Tamper Localization and Copyright Protection </h3>
<h4> CVPR 2024 </h4>
  
[Xuanyu Zhang](https://xuanyuzhang21.github.io/), [Runyi Li](https://villa.jianzhang.tech/people/runyi-li-%E6%9D%8E%E6%B6%A6%E4%B8%80/), [Jiwen Yu](https://vvictoryuki.github.io/website/), [Youmin Xu](https://zirconium2159.github.io/), [Weiqi Li](https://villa.jianzhang.tech/people/weiqi-li-%E6%9D%8E%E7%8E%AE%E7%90%A6/), [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University

[![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)](https://arxiv.org/pdf/2312.08883.pdf)
[![Home Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://xuanyuzhang21.github.io/project/editguard/)
[![CVPR camera-ready](https://img.shields.io/badge/Camera_Ready-<PDF>-red)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_EditGuard_Versatile_Image_Watermarking_for_Tamper_Localization_and_Copyright_Protection_CVPR_2024_paper.pdf)

</div>

## News
- 🎉🎉🎉 The Improved version of EditGuard has been accepted by CVPR 2025. Please check [OmniGuard](https://arxiv.org/abs/2412.01615).
- 🎉🎉🎉 Our code has been released. Our project is still being updated.
- 🎉🎉🎉 Congratulations on EditGuard being accepted by CVPR 2024! Our open-source project is making progress, stay tuned for updates!

## Installation
```
pip install -r requirements.txt
```

## Checkpoint
| Condition | Link |
|------------|------------|
| Clean with better fidelity     | [clean.pth](https://drive.google.com/file/d/1w4e1gpdInAv7Lj_NQ7EGgmMuInyfUYgi/view?usp=sharing)|  
| Support Localization under Gaussian Noise ($\sigma$=0-5), JPEG (Q=70-95), Poission Noise |[degrade.pth](https://drive.google.com/file/d/1fAC2EIrMfPKuQa_DdYdxmUwLBbbsTJXC/view?usp=sharing)| 

Note that EditGuard is mainly used for tamper localization, and its copyright embedding and extraction are only trained on a few degradations such as Gaussian noise, Jpeg, and Poisson noise. If you want to get better robustness, please add more degradations and retrain it.

## Testing
Download the [testing dataset](https://drive.google.com/file/d/1s3HKFOzLokVplXV65Z6xcsBJ9qI91Qfv/view?usp=sharing) and place it in the "./dataset/valAGE-Set" and "./dataset/valAGE-Set-Mask". Download the pre-trained checkpoint "clean.pth" and put it in the "./checkpoints".
```
cd code
python test.py -opt options/test_editguard.yml --ckpt ../checkpoints/clean.pth
```
To extract the tampered masks:
```
python maskextract.py --threshold 0.2
```
To test the localization results under degradation conditions, set the ''degrade_shuffle=True'' in Line 25 of the "options/test_editguard.yml" and download the "degrade.pth".
```
cd code
python test.py -opt options/test_editguard.yml --ckpt ../checkpoints/degrade.pth
python maskextract.py --threshold 0.4
```

## Training
Download the [COCO2017](http://images.cocodataset.org/zips/train2017.zip) dataset and modify the path of the training dataset in the config file.

**Stage 1:** Train the BEM and BRM. 
```
python train_bit.py -opt options/train_editguard_bit.yml
```
**Stage 2:** First modify the checkpoint path of pretrained BEM and BRM in Line 87 "pretrain_model_G: " of "train_editguard_image.yml". Then, please run:
```
python train.py -opt options/train_editguard_image.yml
```

## Introduction

![](./asserts/intro.png)

We propose a versatile proactive forensics framework **EditGuard**. The application scenario is shown on the left, wherein users embed invisible watermarks to their images via EditGuard in advance. If suffering tampering, users can defend their rights via the tampered areas and copyright information provided by EditGuard. Some supported tampering methods (marked in blue) and localization results of EditGuard are placed on the right. Our EditGuard can achieve over **95\%** localization precision and nearly **100\%** copyright accuracy.

## Results

 Our EditGuard can pinpoint pixel-wise tampered areas under different AIGC-based editing methods.

![](./asserts/result.png)

## Extension

Our EditGuard can be easily modified and adapted to video tamper localization and copyright protection.

<table>
  <tr>
    <td colspan="1"><center>Original Video</center></td>
    <td colspan="1"><center>Watermarked Video</center></td>
    <td colspan="1"><center>Tampered Video</center></td>
    <td colspan="1"><center>Predicted Mask</center></td>
  </tr>
  <tr>
    <td><img src="asserts/gif/11.gif" alt="11ori"></td>
    <td><img src="asserts/gif/11_wm.gif" alt="11wm"></td>
    <td><img src="asserts/gif/11_tamper.gif" alt="11tamper"></td>
    <td><img src="asserts/gif/11_mask.gif" alt="11mask"></td>
  </tr>
  <tr>
    <td><img src="asserts/gif/13.gif" alt="13ori"></td>
    <td><img src="asserts/gif/13_wm.gif" alt="13wm"></td>
    <td><img src="asserts/gif/13_tamper.gif" alt="13tamper"></td>
    <td><img src="asserts/gif/13_mask.gif" alt="13mask"></td>
  </tr>
  <tr>
    <td><img src="asserts/gif/tennis.gif" alt="tennisori"></td>
    <td><img src="asserts/gif/tennis_wm.gif" alt="tenniswm"></td>
    <td><img src="asserts/gif/tennis_tamper.gif" alt="tennistamper"></td>
    <td><img src="asserts/gif/tennis_mask.gif" alt="tennismask"></td>
  </tr>
  <tr>
    <td><img src="asserts/gif/umbrella.gif" alt="umori"></td>
    <td><img src="asserts/gif/umbrella_wm.gif" alt="umwm"></td>
    <td><img src="asserts/gif/umbrella_tamper.gif" alt="umtamper"></td>
    <td><img src="asserts/gif/umbrella_mask.gif" alt="ummask"></td>
  </tr>
  <tr>
    <td><img src="asserts/gif/3.gif" alt="tennisori"></td>
    <td><img src="asserts/gif/3_wm.gif" alt="tenniswm"></td>
    <td><img src="asserts/gif/3_tamper.gif" alt="tennistamper"></td>
    <td><img src="asserts/gif/3_mask.gif" alt="tennismask"></td>
  </tr>
  

</table>

## Bibtex
```
@inproceedings{zhang2024editguard,
  title={Editguard: Versatile image watermarking for tamper localization and copyright protection},
  author={Zhang, Xuanyu and Li, Runyi and Yu, Jiwen and Xu, Youmin and Li, Weiqi and Zhang, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11964--11974},
  year={2024}
}
```
