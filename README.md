# [VCNet: a robust approach to blind image inpainting](https://arxiv.org/pdf/2003.06816.pdf) [[Supp]()]
by [Yi Wang](https://shepnerd.github.io/), [Ying-Cong Chen](https://yingcong.github.io/), [Xin Tao](), and [Jiaya Jia](http://jiaya.me/). **The code will be released soon**.

## Introduction
This repository gives the implementation of our method in ECCV 2020 paper, '[VCNet: a robust approach to blind image inpainting](https://arxiv.org/pdf/2003.06816.pdf)'.

<!--
## Framework
We __normalize__ the input feature maps __spatially__ according to the __semantic layouts__ predicted from them. It improves the distant relationship in the input as well as preserving semantics spatially.

<img src="./media/attentive-normalization-frame-v2.png" width="100%" alt="Framework">

Our method is built upon instance normalization (IN). It contains semantic layout learning module (semantic layout prediction + self-sampling regularization) and regional normalization.

<img src="./media/panda.png" width="100%" alt="learned semantic layouts">
<img src="./media/castle.png" width="100%" alt="learned semantic layouts">

The above figure gives the visualization of the learned semantic layout on ImageNet. 1st column: Class-conditional generation results from our method. 2nd column: Binary version of the learned semantic layout. Other columns: Attention maps activated by the learned semantic entities. The brighter the activated regions
are, the higher correlation they are with the used semantic entity.
-->

<!--

## Applications
This module can be applied to the current GAN-based conditional image generation tasks, e.g., class-conditional image generation and image inpainting.
<img src="./media/teaser_v2.gif" width="100%" alt="applications">

In common practice, Attentive Normalization is placed between the convolutional layer and the activation layer. In the testing phase, we remove the randomness in AttenNorm by switching off its self-sampling branch. Thus, the generation procedure is deterministic only affected by the input.

-->




<!--
## Implementation

### Semantic Inpainting
-->

<!--
## Prerequisites
- Python3.5 (or higher)
- PyTorch 1.6 (or later versions) with NVIDIA GPU or CPU
- OpenCV
- numpy
- scipy
- easydict


### Datasets
- Paris-Streetview ([https://github.com/pathak22/context-encoder](https://github.com/pathak22/context-encoder)).


## Disclaimer

## References

-->

## Citation

If our method is useful for your research, please consider citing:

    @article{wang2020vcnet,
        title={VCNet: A Robust Approach to Blind Image Inpainting},
        author={Wang, Yi and Chen, Ying-Cong and Tao, Xin and Jia, Jiaya},
        journal={arXiv preprint arXiv:2003.06816},
        year={2020}
    }

<!--
Our code is built upon [Self-Attention-GAN](https://github.com/heykeetae/Self-Attention-GAN), [SPADE](https://github.com/NVlabs/SPADE), and [Sync-BN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).
-->

### Contact
Please send email to yiwang@cse.cuhk.edu.hk.
