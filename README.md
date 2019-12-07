# `pyhat`
## AM207 Final Project: Implementation of Vehtari et al. 2019
Authors: Tanveer Karim and Ian Weaver

## Introduction
The traditional $\widehat R$ diagnostic to study convergence of MCMC chains proposed by
Gelman and Rubin 1992 and the split-Rhat diagnostic proposed by Gelman et al.
2013 fails to capture convergence issues in many cases. [Vehtari et al.
2019](https://arxiv.org/pdf/1903.08008.pdf) proposes an alternative Rhat metric
that is more robust in capturing mixing problems, especially near the tails. In
this repository, we implement the python version of this Rhat metric. 

In addition, [Vehtari et al. 2019](https://arxiv.org/pdf/1903.08008.pdf)
proposes that rankplots should replace traceplots as visual diagnostics for
MCMC. 
We also implement a python version of rankplot that is able to take pymc3
MultiTrace chain objects as inputs. 

## Organization
`pyhat` is organized into the following utility functions and demonstration
notebooks below:
```
pyhat
├── Project_Summary_Notebook_ian.ipynb
├── Project_Summary_Notebook_tanveer.ipynb
├── README.md
├── codes
│   ├── plotutils.py
│   └── utils.py
└── examples
    ├── multiplanet
    │   ├── data
    │   │   ├── map_solution.npy
    │   │   ├── multiplanet_chain_1.pkl
    │   │   ├── multiplanet_chain_2.pkl
    │   │   ├── multiplanet_chain_3.pkl
    │   │   ├── multiplanet_chain_4.pkl
    │   │   └── trace.pkl
    │   └── multiplanet.ipynb
    ├── rhat_variance
    │   ├── data
    │   │   └── models.npy
    │   └── rhat_variance.ipynb
    └── toymodel_gaussian
        ├── toymodel_2DGaussian.ipynb
        ├── toymodel_gaussian.ipynb
        └── utils.py
```

The main implementation and notebooks detailing its use are:
* [`codes/`](https://nbviewer.jupyter.org/github/icweaver/pyhat/tree/master/codes/) - implementation of modified $\widehat R$ statistics proposed by paper and tools to visualize them
* [`examples/multiplanet/multiplanet.ipynb`](https://nbviewer.jupyter.org/github/icweaver/pyhat/blob/master/examples/multiplanet/multiplanet.ipynb?flush_cache=true) - domain specific application (astronomy) of the tools described above
* [`examples/rhat_variance/rhat_variance.ipynb`](https://nbviewer.jupyter.org/github/icweaver/pyhat/blob/master/examples/rhat_variance/rhat_variance.ipynb?flush_cache=True) - increased variance example
* [`examples/toymodel_gaussian/toymodel_2DGaussian.ipynb`](https://nbviewer.jupyter.org/github/icweaver/pyhat/blob/master/examples/toymodel_gaussian/toymodel_2DGaussian.ipynb?flush_cache=true) - presentation of $\widehat R$ and visualization tools on a 2D Gaussian with changing correlation
* [`examples/toymodel_gaussian/toymodel_gaussian.ipynb`](https://nbviewer.jupyter.org/github/icweaver/pyhat/blob/master/examples/toymodel_gaussian/toymodel_gaussian.ipynb?flush_cache=true) - presentation of $\widehat R$ and visualization tools on a simple Gaussian distribution with changing variance

Note: the `data` folders holds intermediate results that can be loaded into each
notebook to avoid running time-intensive cells again.
`examples/multiplanet/data/trace.pkl` was too large to upload to our Github
repo, but we are more than happy to share it upon request.

**We recommend starting with
[`examples/toymodel_gaussian/toymodel_gaussian.ipynb`](https://nbviewer.jupyter.org/github/icweaver/pyhat/blob/master/examples/toymodel_gaussian/toymodel_gaussian.ipynb?flush_cache=true) because it introduces
and details the implementation of the modified $\widehat R$ diagnostic and
visualization tools proposed by [Vehtari et al.
(2019)](https://ui.adsabs.harvard.edu/abs/2019arXiv190308008V/abstract).**

The write-up for this project is in [`Project_Summary_Notebook.ipynb`](https://nbviewer.jupyter.org/github/icweaver/pyhat/blob/master/Project_Summary_Notebook.ipynb?flush_cache=true)
