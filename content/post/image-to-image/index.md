---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Notes: Image-to-image Translation with CGAN, Isola 2016"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2019-10-06T16:39:07-04:00
lastmod: 2019-10-06T16:39:07-04:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Label image"
  focal_point: "Smart"
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

# Image-to-image Translation with Conditional Adversarial Networks

**Authors**: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros

**Link**: [arXiv](https://arxiv.org/abs/1611.07004)

## 1 Abstract

The authors investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problem. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This work also suggests we can achieve resonable results without hand-engineering our loss functions.

## 2 Introduction

The authors define **automatic image-to-image translation** as the task of tranlating one possible representation of a scene into another, given sufficient training data.

{{% alert note %}}
This is a very interesting task and could be a good idea in my microstructure research:
$$Features \Leftrightarrow Label_image \Leftrightarrow Real_image$$
{{% /alert %}}

**A naive approach:** We ask the CNN to minimize the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results. Coming up with loss functions that force the CNN to output sharp, realistic images is an open problem and generally requires expert knowledge.

**With GANs:** It would be highly desirable if we could instead specify only a high-level goal, like "make the output indistinguishable from reality", and then automatically learn a loss function appropriate for satisfying this goal, which is exactly what is done by the GANs.

**Contributions:** The primary contribution of this paper is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results. The second contribution is to present a simple framework sufficient to achieve good results, and to analyze the effects of several important architectual choices. See published codes [here](https://github.com/phillipi/pix2pix).

## 3 Related work

**Structured losses for image modeling:** Image-to-image translation problems are often formulated as per-pixel classification or regression. The output space are "unstructured" in the sense that each output pixel is considered conditionally independent from all others given the input image. Conditional GANs instead learn a "structured" loss.

## 4 Method

Conditional GANs learn a mapping from observed image $x$ and random noise $z$, to $y$, $G: \{x, z\} \to y$.

### 4.1 Objective

The objective of a conditional GAN can be expressed as

$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}\left[ \log D(x,y) \right] + \mathbb{E}_{x,z} \left[ \log(1 - D(x, G(x, z))) \right]$$

where $G^* = \arg \min_G \max_D \mathcal{L}_{cGAN}(G, D)$.

To test the importance of conditioning the discriminator, we also compare to an unconditional variant in which the discriminator does not observe $x$:

$$\mathcal{L}_{GAN}(G, D) = \mathbb{E}_y\left[ \log D(y) \right] + \mathbb{E}_{x, z} \left[ \log (1 - D(G(x,z))) \right]$$

Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss. Here we use L1 distance rather than L2 as L1 encourages less blurring:

$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z} \lVert y - G(x, z) \rVert_1$$

The final objective is

$$G^* = \arg \min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

In initial experiments, the generator simply learned to ignore the noise. In the final models, the authors provided noise only in the form of dropout, applied on several layers of the generator at both training and test time. Depsite the dropout noise, we observe only minor stochasticity in the outputs. Designing conditional GANs that produce highly stochastic output, and thereby capture the full entropy of the conditional distributions they model, is an important questions left open by the present work.

{{% alert note %}}
A very common issue plagued my GANs.
{{% /alert %}}

### 4.2 Network architectures

The generator and discriminator architectures follow the DCGAN design and use modules of the form convolution-BatchNorm-ReLU.

**Generator with skips:** For many image translation problems, there is a great deal of low-level information shared between the input and output, and it would be desirable to shuttle this information directly across the net. We add skip connections, following the general shape of a "U-net". Sepcifically, we add skip connections between each layer $i$ and layer $n-i$.

{{% alert note %}}
This skip connection idea is also used in my cancer image segmentation research. I don't know much about the cancer tissues but I think we want to retain both high-level and low-level textures of a tissue image.
{{% /alert %}}]

**Markovian discriminator (PatchGAN)**: It is *well-known* that the L2 loss - and L1 - produces blurry results on image generation problems. This motivates restricting GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness. The authors restrict the attention to the structure in local image patches and design a discriminator architecture - which they term a PatchGAN - that only penalizes structure at the scale of patches. This discriminator tries to classify if each $N \times N$ patch in an image is real or fake. Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. Therefore, the PatchGAN can be understood as a form of texture/style loss.

{{% alert warning %}}
What if we can model the textures/styles and generate the image from the label?
{{% /alert %}}

### 4.3 Optimization and inference

A few implementation details to optimize the networks:

1. Alternate between 1 gradient descent on $D$, then one step on $G$ (as in the original GAN paper).
2. Rather than training $G$ to minimize $\log(1 - D(x, G(x,z)))$, we train to maximize $\log D(x, G(x, z))$ instead.
3. Divide the objective by 2 while optimizing $D$, which slows down the rate at which $D$ learns relative to $G$.
4. Use minibatch SGD and apply the Adam solver, with a learning rate of 0.0002, and momentum parameters $\beta_1 = 0.5$, $\beta_2 = 0.999$.
