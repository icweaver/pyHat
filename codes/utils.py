import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pymc3 as pm
import pandas as pd
import pickle

def rhat(trace, param=None, split=True):
    if isinstance(trace, pm.backends.base.MultiTrace):
        # separate chains
        theta = np.array(trace.get_values(param, combine=False))
    elif isinstance(trace, np.ndarray):
        theta = trace
    else:
        raise Exception("Trace must by pymc3 object or ndarray")

    if split: # split chains in half and stack
        theta = np.r_[tuple(np.hsplit(theta, 2))]

    M = theta.shape[0] # number of chains
    N = theta.shape[1] # number of samples per chain

    # between-chain variance (eq. 1)
    theta_hat_dotm = lambda m, N=N: (1./N)*np.sum(theta[m])
    theta_hat_ddot = (1./M)*np.sum(np.mean(theta, axis=1))
    theta_hat_diff = [theta_hat_dotm(m) - theta_hat_ddot for m in range(M)]
    B = (N/(M - 1.))*np.sum((np.array(theta_hat_diff))**2)

    # within-chain variance (eq. 2)
    theta_diff = lambda m: theta[m] - np.mean(theta[m])
    s_m_sq = lambda m, N=N: (1./(N - 1.))*np.sum(theta_diff(m)**2)
    W = (1./M)*np.sum([s_m_sq(m, N=N) for m in range(M)])

    # marginal posterior variance estimate (eq. 3)
    post_var = ((N - 1.)/N)*W + (1./N)*B

    # rhat (eq. 4)
    return np.sqrt(post_var / W)

def rank(x):
        """computes ranks for a 1D array using
        the average method to break ties."""
        import pandas as pd

        r = pd.DataFrame(x).rank().values[:,0] 

        if(len(x) != 0): #make sure that x has at least 1 element
                return r
        else:
                raise Exception("You gave a 0 length array.")

def zscale(chains):
    """
    NDarray object for M chains for a specific parameter
    """
    S = len(chains.flatten())
    r = rank(chains.flatten())
    z = sp.stats.norm.ppf((r - 0.5) / S).reshape(chains.shape)
    return z

def rank_rhat(chains):
    z = zscale(chains)
    return rhat(z, split=True)

def folded_split_rhat(trace):
    zeta = np.abs(trace - np.median(trace).reshape(-1, 1))
    zscale_folded = zscale(zeta)
    return rhat(zscale_folded, split = True)

def savetrace(fname, trace, model):
        """Saves trace and model in pickle format.
        Parameters
        ----------
        fname : name of file to be saved
                string
        trace : trace output
                pymc3 trace
        model : model output
                pymc3 model"""
        import pickle

        with open(fname, 'wb') as buff:
                pickle.dump({'model': model, 'trace': trace}, buff)

def loadtrace(fname):
        """Loads trace and model from pickle file."""
        import pickle

        with open(fname, 'rb') as buff:
                data = pickle.load(buff)  

        model, trace = data['model'], data['trace']

        return model, trace

def invPhi(y, mu = 0., sigma = 1.):
    return mu + sigma*np.sqrt(2*np.log(1/(sigma*np.sqrt(2*np.pi)*y)))

def print_percent_diff(x_n, x_0):
    percent_diff = (np.abs(x_n - x_0) / x_0)*100.
    print(f"percent difference: {percent_diff:.2f}%")

def rhat_results(d):
    #
    # returns table of rhat values for parameter in dict (d)
    #
    index = [
        "standard",
        "split",
        "ranked",
        "folded"
    ]
    df_rhats = pd.DataFrame(index=index)
    for param_name, param_chains in d.items():
        rhat_standard = rhat(param_chains, split=False)
        rhat_split = rhat(param_chains, split=True)
        rhat_rank = rank_rhat(param_chains)
        rhat_folded = folded_split_rhat(param_chains)
        df_rhats[param_name] = [rhat_standard, rhat_split, rhat_rank, rhat_folded]
        
    return df_rhats
