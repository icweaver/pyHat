# pyhat
## AM207 Final Project: Implementation of Vehtari et al. 2019
Authors: Tanveer Karim and Ian Weaver

## Introduction
The traditional Rhat diagnostic to study convergence of MCMC chains proposed by Gelman and Rubin 1992 and the split-Rhat diagnostic proposed by Gelman et al. 2013 fails to capture convergence issues in many cases. [Vehtari et al. 2019](https://arxiv.org/pdf/1903.08008.pdf) proposes an alternative Rhat metric that is more robust in capturing mixing problems, especially near the tails. In this repository, we implement the python version of this Rhat metric. 

In addition, [Vehtari et al. 2019](https://arxiv.org/pdf/1903.08008.pdf) proposes that rankplots replace traceplots as visual diagnostics for MCMC. We also implement a python version of rankplot that is able to take pymc3 MultiTrace chain objects as inputs. 

All the necessary functions can be found in codes/utils.py. All the necessary plotting functions can be found in codes/plotutils.py. We include a toy model problem in examples/toymodel_gaussian.ipynb.
