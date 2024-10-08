# FMImaging - Foudation Model for Clinical and Biomedical Imaging

Main contributors: **Hui Xue, Sarah Hoopper, Azaan Rehman, Chris Combs, Peter Kellman**

## References

- [Imaging transformer for MRI denoising with the SNR unit training: enabling generalization across field-strengths, imaging contrasts, and anatomy](https://arxiv.org/abs/2404.02382)

```
@misc{xue2024imaging,
      title={Imaging transformer for MRI denoising with the SNR unit training: enabling generalization across field-strengths, imaging contrasts, and anatomy}, 
      author={Hui Xue and Sarah Hooper and Azaan Rehman and Iain Pierce and Thomas Treibel and Rhodri Davies and W Patricia Bandettini and Rajiv Ramasawmy and Ahsan Javed and Zheren Zhu and Yang Yang and James Moon and Adrienne Campbell and Peter Kellman},
      year={2024},
      eprint={2404.02382},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
- [Characterizing the Signal-to-noise Ratio and Spatio-temporal Resolution of an Imaging Transformer Model for CMR](https://www.journalofcmr.com/article/S1097-6647(24)00974-8/fulltext)
```
https://doi.org/10.1016/j.jocmr.2024.100983
```
-(Convolutional Neural Network Transformer (CNNT) for Fluorescence Microscopy image Denoising with Improved Generalization and Fast Adaptation)[https://arxiv.org/abs/2404.04726]
```
@misc{rehman2024convolutional,
      title={Convolutional Neural Network Transformer (CNNT) for Fluorescence Microscopy image Denoising with Improved Generalization and Fast Adaptation}, 
      author={Azaan Rehman and Alexander Zhovmer and Ryo Sato and Yosuke Mukoyama and Jiji Chen and Alberto Rissone and Rosa Puertollano and Harshad Vishwasrao and Hari Shroff and Christian A. Combs and Hui Xue},
      year={2024},
      eprint={2404.04726},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```

## Background

The established imaging modalities (e.g. MRI, CT, US or Microscopy etc.) serve as a foundation to develop deep learning applications for imaging AI. The diversity in modalities, anatomies, imaging experiment setups, and downstream tasks poses challenges. Conventional deep learning workflows require application-specific data collection, model training, and deployment. Such application-specific AI workflows are difficult to scale up for new imaging tasks, as collecting data and generating labels are costly. Further, data samples acquired for one task may not help another task, by only training the task specific model from the scratch. This is one major barrier to develop broader imaging AI, as there are many different tasks, but datasets for every task is either to-be-curated or available but small. 

With the recent success in large language model for multi-task generalization, it is a question to ask whether imaging AI can enjoy the similar success. This project aims to develop foundation model for clinical and biomedical imaging and evaluate its generalization performance on new tasks, especially with limited labeled data.

Aims:

- Develop and train backbone models on large imaging datasets
- Develop methods to quickly adapt the backbone model to new imaging tasks, with reduced data labeling effort
- Develop and deploy front-end and backbone models to enable user-driven imaging AI development
- Focus more on pixel-level predictions tasks, such as restoration and reconstruction, to recover signal from low SNR or incomplete acquisition, with high fidelity

## Data formation

Unlike the ImageNet type images, the clinical or biomedical imaging have more variation in image formation. The imaging can acquire 2D, 2D+T dynamic or 3D images. Rarely, 3D+T dynamic volume imaging can be performed as well. At the minimum, we will handle 2D, 2D+T and 3D images. We propose an uniform model to process three image formation as the tensor [B, C, T/SLC, H, W]. The 2nd dimension can be temporal for 2D+T or SLC or Z direction for 3D volume. For the 2D images, the T/SLC will be 1. C is the channel; H and W are height and weight.

```
Tensor format: [B, C, T/SLC, H, W]
```

## Hypothesis

The transformer type models can learn broad knowledge and generalize across multiple downstream tasks. In the LLMs, it can lead to emergent abilities with few-shot in-context learning. Here we hypothesize modified transformer models can learn general knowledge of anatomy, imaging and pathology and be used as general-purpose backbone.

<img src="./doc/images/background.JPG"  width="40%" height="30%">

## Modules for the foundation model

### Local, global and temporal attention

Instead of cutting images into patches and process them with the standard transformer, we propose to explore both spatial and temporal/slice correlation with three attention mechanisms. 

<img src="./doc/images/stcnnt_illustration.JPG"  width="50%" height="48%">

Let the image size be $[H, W]$, the windows size be $[w, w]$ and patch size be $[k, k]$. The number of windows will be $[\frac {H}{w}, \frac{W}{w}]$. The number of patches is $[\frac{H}{k}, \frac{W}{k}]$. For example, for an $256 \times 256$ images, $w=32$ and $k=16$, we will have $8 \times 8$ windows and each window will have $2 \times 2$ patches.

**Local spatial attention (L)** 
Local attention is computed by attening to the neighboring pixels in images or feature maps. The feature vectors at all red locations (key and values, K and V) are compared towards the yellow vector (Q, query) to get the attention coefficients. The attention outputs for yellow pixel is a weighted sum of value vectors.This is the same operation as the [swin transformer](https://arxiv.org/abs/2103.14030).

The local attention computes a new pixel by computing attention coefficients of all patches in a window. The local attention matrix is $\frac {w}{k} \times \frac {w}{k}$.

**Global spatial attention (G)**
While the local attention only explores neighboring pixels, global attention looks at more remote pixels for correlation. This will help model learn global information over larger field-of-view. In the global attention, the blue vectors serve as K and V. The yellow vector, as Q, will compute its attention to K vectors. 

With a stride $[S, S]$, the global attention computes a new patch by computing attention coefficients of $w^2$ windows together. All corresponding patches in all windows are attened to each other. So for every patch, attention matrix is $w^2 \times w^2$. The difference is that patches are sampled with a stride.

<img src="./doc/images/attention_in_details.jpg"  width="50%" height="100%">

**Local patch attention (LP)**
First the image is split to windows. All patches within a window are inputs for attention. The tensor is reshaped to $[B, T, \frac{H}{w}, \frac{K}{w}, \frac{w}{k}, \frac{w}{k}, k \times k \times C]$. $\frac{w}{k} \times \frac{w}{k}$ is the number of patches in a window. The attention is computed among patches within one image. Attention matrix size is $[\frac{w}{k} \frac{w}{k} \times \frac{w}{k} \frac{w}{k}]$.

**Global patch attention (GP)**
In the global attention, all corresponding patches from all windows are inputs for attention. The number of windows is $\frac{H}{w} \times \frac{W}{w}$. Every window has $\frac{w}{k} \times \frac{w}{k}$ patches. The tensor is reshaped to $[B, T, \frac{w}{k}, \frac{w}{k}, \frac{H}{w}, \frac{W}{w}, k \times k \times C]$. The attention matrix size is $\frac{H}{w} \frac{W}{w} \times \frac{H}{w} \frac{W}{w}$.

**ViT attention (V)**
Besides the pixel-wise local or global attention, we also implement the ViT type attention. First, the image is split into consecutive, non-overlapped $[k, k]$ patches by reshaping the $[B, C, T, H, W]$ to $[B, T, \frac{H}{k}, \frac{W}{k}, C \times k \times k]$. Then the $[\frac{H}{k}\frac{W}{k} \times \frac{H}{k}\frac{W}{k}]$ attention matrix is computed for every $B$ and $T$. ViT attention is a type of global attention, while a window has only one patch. Its computation grows quardratically as to the number of patches in an image.

**Conv vs. Linear**
To implement local and global attention, linear Q/K/V parameter matrixes can be utilized in the attention mechanism:

$$Q = W_q F, K = W_k F, V = W_v F$$

where the parameter matrixes are $W_q, W_k, W_v$. $F$ is the flattened feature vectors.

The computation increases quadratically as the number of pixels in the attention window. However, the patch size cannot be too small (otherwise, all pixels in a window can be too similar and attention will not be effective).

Alternatively, we can parameterize the Q/K/V computation with convolution:

$$Q = {CONV}_q (F), K = {CONV}_k (F), V = {CONV}_v (F)$$

Here $CONV$ is the convolution over [C, H, W] for the pixels in the window and $F$ is the un-flattened feature maps.

Compare to the linear matrixes, the $CONV$ keeps the inductive bias and significantly reduce the computational cost. For the global attention, this is equivalent to dilated convolution.

**Temporal attention (T)**
The temporal or slice correlation is explored by computing the temporal attention. Given $T$ images or feature maps in a tensor $[B, C, T, H, W]$, the attention is computed between each $[C, H, W]$ array, resulting to a $T \times T$ attention matrix. Given the very high number of pixels in feature maps, the $CONV$ is used to compute $Q/K/V$. 

Another way (**T-pixel**) is to reshape the tensor to be $[B, H, W, T, C]$. For every pixel, a $T \times T$ attention matrix is computed.

These **L, G, T** attention mechanisms are implemented as **Attention** modules.

### Cell

After implementing each attention module, a **Cell** is defined as one **Attention** module with the **mixer**. Given an input tensor $[B, T, C_{in}, H, W]$, a Cell transforms the input tensor and produce another tensor $[B, T, C_{out}, H, W]$.

<img src="./doc/images/Cell.JPG"  width="80%" height="60%">

Two cell architectures are implemented: **sequential** and **parallel**.

ref for parallel: [here](https://arxiv.org/abs/2302.05442). 

Different $Norm$ operations are supported: 

| Norm | Description |
| ----------- | ----------- |
| LayerNorm | normalize over [C, H, W] |
| BatchNorm 2D | normalize over [H, W], B*T are batch dimension |
| BatchNorm 3D | normalize over [T, H, W] |
| InstanceNorm 2D | normalize over [H, W]|
| InstanceNorm 3D | normalize over [T, H, W]|

Except the layernorm, all other norms support variable sized images, when using with the $CONV$ in attention layers.

The $Mixer$ is implemented as convolution 2D operations, along the $[C, H, W]$:

```
self.mlp = nn.Sequential(
                Conv2DExt(C_out, 4*C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.GELU(),
                Conv2DExt(4*C_out, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.Dropout(dropout_p),
            )
```

User can specify whether a cell has mixer or not.

We term this spatio-temporal CNN transformer for imaging as **ST-CNNT**. 

### Block
A block is a stack of cells. For example, a block with local, global and temporal attentions can be built as:

![Block with three attentions](./doc/images/Block.JPG)

This block is coded as the block string "L1G1T1". The letter "L", "G" or "T" means an attention layer. "1" means the mixer is added on top of the attention mechanism (if "0", mixer is not added; we can have a stack of attention only layers). As an example, "L0L1G0G1T1" means a block with 5 attention layers. Mixers are not added to the first and third attentions, but added after the second and fourth attentions. The last cell is a temporal attention with its mixer added.

This block string method gives a good amount of flexibility to assemble and test different attention configurations. Depending on the image resolutions and number of feature maps, different blocks in a model can have different attention configuration.

To compare model architectures, the block string can be "C2" or "C3". "C2" means the 2D conv is used in the cell. "C3" means the 3D conv is used (over T, H, W). By changing the block string from attentions to convs, we can get identical architectures with only differences being the cell structure.

## Backbone models
How to put together attention modules to make a model? How many attentions will we need? In what architecture? 

In the LLMs, the stack of attentions proves to be very effective. For imaging, previous researches had explored similar architecture (e.g. [3B swin v2](https://arxiv.org/abs/2111.09883) and [20B ViT](https://arxiv.org/abs/2302.05442)). With our intention to combine convolution back into the transformer and to utilize the inductive bias, we should explore different architectures.

### ST-CNNT U-Net

The U-net architecture is enhanced with ST-CNNT cells. 

<img src="./doc/images/stcnnt_Unet.JPG"  width="30%" height="100%">

Here every resolution stage includes one block containing multiple cells. Model can specify number of feature maps at each resolution stage. The [Unet with attention](https://arxiv.org/abs/1804.03999) is implemented here. Downsample and upsample are implemented with interpolation.

The [attention in unet](https://arxiv.org/abs/1804.03999) is added to gate the feature map in lower resolution level to guide the higher resolution.

<img src="./doc/images/Attention_in_Unet.JPG"  width="50%" height="60%">

### ST-CNNT HR-Net

This network is modified from the [high-resolution architecture](https://www.microsoft.com/en-us/research/blog/high-resolution-network-a-universal-neural-architecture-for-visual-recognition/).


![stcnnt_hrnet](./doc/images/stcnnt_HRNet.JPG)

The network is defined as levels and stages. Every block is numbered by its level and stage indexes (starting from 0). The downsample and upsample modules are added to link different blocks. Different up/downsample modules are implemented, with TLG attentions or 1x1 CONV. Bilinear interpolation is used to alter spatial resolution.

After the fusion stage, the model will output per-level tensors and the aggregated tensor. 

### ST-CNNT LLM

As a baseline, the stack of blocks serves as an architecture similar to the LLMs. The difference is the non-causal attention is used in all blocks and we added the dense skip connections between blocks (default, it is off).

<img src="./doc/images/stcnnt_LLM.JPG"  width="20%" height="100%">

In the current design, the Block will not change number of input channels. 

### Other architectures

#### Swin 3D
A swin 3D implementation is added to backbone_omnivore.py. It implements the swin 3D backbone with the downsampling layers removed. 

--

# Refactor notes, v1

## Overview
This is the v1 refactor of the codebase, which contains basic utilities for single-task training with DDP. The purpose of the refactor was to:
  * Reduce the amount of new code needed for new projects, with zero-code solutions for basic applications and lightweight customizations for others.
  * Reduce the amount of rewritten code per project.
  * Make organization clearer (e.g., by consolidating configs, metrics, optimizers, losses, etc.) to keep codebase clean as we continue to add complexity.
  * Prepare consolidated codebase for FM experiments, including:
    * Build in pre/post/backbone structure.
    * Include utils for segmentation, classification, and enhancement tasks.

### Organization
The codebase organizes directories by utility. The ```run.py``` file shows how the codebase progresses:
  * In the ```setup``` dir, args are parsed into a config and initial setup functions are run.
  * In the ```data``` dir, torch datasets are created.
  * In the ```loss``` dir, the loss function is defined.
  * In the ```model``` dir, the Model Manager is defined, which contains the pre/backbone/post modules and utils for saving and loading the model.
  * In the ```optim``` dir, the Optim Manager is defined, which contains the optimizer and scheduler.
  * In the ```metrics``` dir, the Metric Manager is defined, which tracks all performance metrics during training.
  * In the ```trainer``` dir, the Train Manager is defined, which controls DDP and the train/eval loops.

Each project can be stored in the ```projects``` dir. 

