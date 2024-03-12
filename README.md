# SatMAE++: Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery (CVPR 2024)

## Updates
- **March 11, 2024:** SatMAE++ paper is released [[arXiv]](https://arxiv.org/abs/2403.05419)  [[PDF]](https://arxiv.org/pdf/2403.05419.pdf)
- **Code will be released soon ....**

## Overview
Different from standard natural image datasets, remote sensing data is acquired from various sensor technologies and exhibit diverse range of scale variations as well as modalities. Existing satellite image pre-training methods either ignore the scale information present in the remote sensing imagery or restrict themselves to use only a single type of data modality. Compared to existing works, SatMAE++ with multi-scale pre-training is equally effective for both optical as well as multi-spectral imagery. SatMAE++ performs multi-scale pre-training and utilizes convolution based upsampling blocks to reconstruct the image at higher scales making it extensible to include more scales.

## Method
SatMAE++ incorporates the multiscale information by reconstructing the image at multiscale levels thereby improving the performance on various scene classification downstream datasets.

<img width="1096" alt="image" src="images/overall_architecture.png">


