# Resolution Attack: Exploiting Image Compression to Deceive Deep Neural Networks

---

**Authors:**  
Wangjia Yu<sup>1,2</sup>, Xiaomeng Fu<sup>1,2</sup>, Qiao Li<sup>1,2</sup>, Jizhong Han<sup>1</sup>, Xiaodan Zhang<sup>1*</sup>

**Institutional Information:**  
Institute of Information Engineering, Chinese Academy of Sciences<sup>1</sup>  
School of Cyber Security, University of Chinese Academy of Sciences<sup>2</sup>  
{yuwangjia,fuxiaomeng,liqiao,hanjizhong,zhangxiaodan}@iie.ac.cn

This repository contains the official implementation for **Resolution Attack: Exploiting Image Compression to Deceive Deep Neural Networks**. [Read the full paper here](https://openreview.net/pdf?id=OFukl9Qg8P).

**Abstract:**  
---

Model robustness is essential for ensuring the stability and reliability of machine learning systems. Despite extensive research on various aspects of model robustness, such as adversarial robustness and label noise robustness, the exploration of robustness towards different resolutions, remains less explored. To address this gap, we introduce a novel form of attack: the resolution attack. This attack aims to deceive both classifiers and human observers by generating images that exhibit different semantics across different resolutions. To implement the resolution attack, we propose an automated framework capable of generating dual-semantic images in a zero-shot manner. Specifically, we leverage large-scale diffusion models for their comprehensive ability to construct images and propose a staged denoising strategy to achieve a smoother transition across resolutions. Through the proposed framework, we conduct resolution attacks against various off-the-shelf classifiers. The experimental results exhibit high attack success rate, which not only validates the effectiveness of our proposed framework but also reveals the vulnerability of current classifiers towards different resolutions. Additionally, our framework, which incorporates features from two distinct objects, serves as a competitive tool for applications such as face swapping and facial camouflage.

**Intro:**
---


