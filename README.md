# Q-Newton

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper "Hybrid Quantum-Classical Scheduling for Accelerating Neural Network Training with Newton's Gradient Descent"

* Authors: [Pingzhi Li](https://pingzhili.github.io/), [Junyu Liu](https://sites.google.com/view/junyuliu/main), [Hanrui Wang](https://hanruiwang.mit.edu/), and [Tianlong Chen](https://tianlong-chen.github.io/)
* Paper: [arXiv](https://arxiv.org/abs/2405.00252)


## Overview

Optimization techniques in deep learning are predominantly led by first-order gradient methodologies, such as SGD (Stochastic Gradient Descent). However, neural network training can greatly benefit from the rapid convergence characteristics of second-order optimization. Newton's GD (Gradient Descent) stands out in this category, by rescaling the gradient using the inverse Hessian. Nevertheless, one of its major bottlenecks is matrix inversion, which is notably time-consuming in $\mathcal{O}(N^3)$ time with weak scalability. 

Matrix inversion can be translated into solving a series of linear equations. Given that quantum linear solver algorithms (QLSAs), leveraging the principles of quantum superposition and entanglement, can operate within a $\mathtt{polylog}(N)$ time frame, they present a promising approach with exponential acceleration. Specifically, one of the most recent QLSAs demonstrates a complexity scaling of $\mathcal{O}(d\cdot\kappa \log(N\cdot\kappa/\epsilon))$, depending on: {size $N$, condition number $\kappa$, error tolerance $\epsilon$, quantum oracle sparsity $d$} of the matrix. However, this also implies that their potential exponential advantage may be hindered by certain properties ($\textit{i.e.}$ $\kappa$ and $d$). For example, a $10^4\times 10^4$ dense matrix with $\kappa$ of $10^4$ may not be inverted more efficiently using QLSAs compared to classical LU decomposition solvers.

We propose $\texttt{Q-Newton}$, a hybrid quantum-classical scheduler for accelerating neural network training with Newton's gradient descent. $\texttt{Q-Newton}$ utilizes a streamlined scheduling module that coordinates between quantum and classical linear solvers, by estimating \& reducing $\kappa$ and constructing $d$ for the quantum solver. Specifically, $\texttt{Q-Newton}$ consists of: 1️⃣ A robust yet lightweight Hessian condition number estimator, derived from its norm and determinant. 2️⃣ A magnitude-based, symmetry-aware pruning method designed to build quantum oracle sparsity and the Hessian regularization that lowers its condition number. They present an effective trade-off between time cost and performance. 3️⃣ A plug-and-play scheduler for Hessian inversion tasks in Newton's GD training process. It dynamically allocates tasks between classical and quantum solvers based on efficiency.

Our evaluation showcases the potential for $\texttt{Q-Newton}$ to significantly reduce the total training time compared to commonly used optimizers like SGD, across a range of neural network architectures and tasks. We hypothesize a future scenario where the gate time of quantum machines is reduced, possibly realized by attoseconds physics. Our evaluation establishes an ambitious and promising target for the evolution of quantum computing.


## Citation

```bibtex
@misc{li2024hybrid,
      title={Hybrid Quantum-Classical Scheduling for Accelerating Neural Network Training with Newton's Gradient Descent}, 
      author={Pingzhi Li and Junyu Liu and Hanrui Wang and Tianlong Chen},
      year={2024},
      eprint={2405.00252},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
