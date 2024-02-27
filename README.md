<div align="center">
<img src="./asserts/Logo.png" alt="Image Alt Text" width="150" height="150">
<h3> EditGuard: Versatile Image Watermarking for Tamper Localization and Copyright Protection </h3>
  
[Xuanyu Zhang](https://xuanyuzhang21.github.io/), [Runyi Li](https://villa.jianzhang.tech/people/runyi-li-%E6%9D%8E%E6%B6%A6%E4%B8%80/), [Jiwen Yu](https://vvictoryuki.github.io/website/), [Youmin Xu](https://zirconium2159.github.io/), [Weiqi Li](https://villa.jianzhang.tech/people/weiqi-li-%E6%9D%8E%E7%8E%AE%E7%90%A6/), [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University

[![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)](https://arxiv.org/pdf/2312.08883.pdf)
[![Home Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://xuanyuzhang21.github.io/project/editguard/)

</div>

## News
- **_News (2024-02-27)_**: 🎉🎉🎉 Congratulations on EditGuard being accepted by CVPR 2024! Our open-source project is making progress, stay tuned for updates!

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

## Code

Our code is coming soon...

## Contact us

xuanyuzhang21@stu.pku.edu.cn

## Bibtex
```
@article{zhang2023editguard,
  author    = {Xuanyu Zhang and Runyi Li and Jiwen Yu and Youmin Xu and Weiqi Li and Jian Zhang},
  title     = {EditGuard: Versatile Image Watermarking for Tamper Localization and Copyright Protection},
  journal   = {arXiv preprint arxiv:2312.08883},
  year      = {2023},
}
```
