---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Notes: Generative Adversarial Nets, Goodfellow 2014"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2019-10-05T13:50:43-04:00
lastmod: 2019-10-05T13:50:43-04:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

# Generative Adversarial Nets

**Authors**: Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Benjio

**Link**: [arXiv](https://arxiv.org/abs/1406.2661)

## 1 Abstract

The authors proposed a new framework for estimating generative models via an adversarial process, in which two models are trained simultaneously: a generative model $G$ that captures the data distribution, and a discriminator model $D$ that estimates the probability that a sample came from the training data rather than $G$. In the space of arbitrary functions $G$ and $D$, a unique solution exists, with $G$ recovering the training data distribution and $D$ equals to $\frac{1}{2}$ everywhere.

{{% alert warning %}}
Since the *target* is the state where the counterfeits are indistinguishable from the genuine, a potential assumption is that the training data do generalize the target data distribution well. This was a problem in my previous research where data are limited.
{{% /alert %}}

## 2 Adversarial nets

To learn the generator's distribution $p\_g$ over data $\mathbf{x}$, we define a prior on input noise variables $p\_\mathbf{z}(\mathbf{z})$, then represent a mapping to data space as $G(\mathbf{z}; \theta_g)$, where $G$ is a differentiable function represented by a multilayer perceptron with parameters $\theta_g$. We also define a second multilayer perceptron $D(\mathbf{x}; \theta_d)$ that outputs a single scalar, which represents the probability that $\mathbf{x}$ came from the data rather than $p_g$. We train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$. We simultaneously train $G$ to minimize $\log(1 - D(G(\mathbf{z})))$. In other words, $D$ and $G$ play the following **two-player minimax game** with value function $V(G, D)$:

$$\min\_G \max\_D V(D, G) = \mathbb{E}\_{\mathbf{x} \sim p\_{data}(\mathbf{x})} \left[ \log D(\mathbf{x}) \right] + \mathbb{E}\_{\mathbf{z} \sim p\_{\mathbf{z}}(\mathbf{z})} \left[ \log D(1 - D(G(\mathbf{z}))) \right]$$

Some problems in practice:

1. Optimizing $D$ to completion in the inner loop of training is computationally prohibitive, and on finite dataset would result in overfitting. Instead, we alternate between $k$ steps of optimizing $D$ and one step of optimizing $G$.
2. The equation above may not provide sufficient gradient for $G$ to learn well. Rather than training $G$ to minimize $\log(1 - D(G(\mathbf{z})))$ we can train $G$ to maximize $\log D(G(\mathbf{z}))$. This objective function provides much stronger gradients early in learning.

## 3 Theoretical Results

> Algorithm 1: Minibatch stochastic gradient descent training of GAN. $k$ is a hyperparameter.
> ```py
> for i in range(num_iterations):
>     for j in range(k):
>         Sample minibatch of m noises from noise distribution
>         Sample minibatch of m examples from data distribution
>         Update the discriminator by ascending its stochastic gradient
>     Sample minibatch of m noises from noise distribution
>     Update the generator by descending its stochastic gradient
> ```
> To update the discriminator, we use
> $$\nabla\_{\theta\_d} \frac{1}{m} \sum\_{i=1}^m \left[ \log D\left(x^{(i)}\right) + \log \left( 1 - D\left( G\left( z^{(i)} \right) \right) \right) \right]$$
> To update the generator, we use
> $$\nabla\_{\theta\_g} \frac{1}{m} \sum\_{i=1}^m \log \left( 1 - D\left( G\left( z^{(i)} \right) \right) \right)$$

### 3.1 Global Optimality of $p_g = p_{data}$

For any given $G$, consider the optimal discriminator $D$.

**Proposition 1.** For $G$ fixed, the optimal discriminator $D$ is
$$D\_G^*(x) = \frac{p\_{data}(x)}{p\_{data}(x) + p\_g(x)}$$

**Theorem 1.** The global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p\_g = p\_{data}$. At that point, $C(G)$ achieves the value $-\log 4$.

### 3.2 Convergence of Algorithm 1

**Proposition 2.** If $G$ and $D$ have enough capacity, and at each step of Algorithm 1, the discriminator is allowed to reach its optimum given $G$, and $p_g$ is updated so as to improve the criterion
$$\mathbb{E}\_{x\sim p\_{data}} \left[ \log D_G^\*(x) \right] + \mathbb{E}\_{x \sim p_g} \left[ \log\left( 1 - D_G^\*(x) \right) \right]$$
then $p\_g$ converges to $p\_{data}$.

## 4 Advantages and disadvantages

Disadvantages:

1. no explicit representation of $p_g(x)$.
2. $D$ must be synchronized well with $G$ during training, in order to avoid "the Helvetica scenario", in which $G$ collapses too many values of $z$ to the same value of $x$ to have enough diversity to model $p_{data}$

{{% alert note %}}
The second disadvantage is a very interesting topic (becuase it is intractable when the data space is large, for example, 1024x1024 images).

This problem has plagued my research for month. In the next stage, I will dig into the problems and find out some general techniques to *avoid* this.
{{% /alert %}}

Advantages:

1. Markov chains are never needed.
2. No inference is needed during learning.
3. A wide variety of functions can be incorporated into the model.

*See the original paper for a detailed comparison between GAN and other generative modeling approaches.*

## 5 Conclusions

1. A conditional generative model $p(x \mid c)$ can be obtained by adding $c$ as input to both $G$ and $D$.
2. Learned approximate inference can be performed by training an auxiliary network to predict $z$ given $x$.
3. One can approximately model all conditionals $p(x\_S \mid x\_{\not S})$ where $S$ is a subset of the indices of $x$ by training a family of conditional models that share parameters.
4. Semi-supervised learning: improve performance of classifiers when limited labeled data is available.
5. Efficiency improvements.

{{% alert note %}}
For the second conclusion: This process is much alike a feature extraction after the convergence of the GAN. Could be a good idea for my feature leanring.

For the third conclusion: The implication sounds exciting but can be a little too hard to be impractical, especially for my project.
{{% /alert %}}
