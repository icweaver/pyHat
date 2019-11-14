import numpy as np
import pymc3 as pm
import pandas as pd
import pickle

def rhat(trace, param, split=True):
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

def S_eff(x, param):
    """ Returns estimate of effective sample size for a set of traces.
        
        Parameters
        ----------
        x : ndarray
            M X N array of traces, where M is the number of chains
            and N is the number of samples.
        param : str
            The parameter that is traced in `x` .
        
        Returns
        ------- 
        S : int
            The effective sample size.
           
        NOTES
        ----- 
        This algorithm is adapted from the `pymc3` source code:
        https://github.com/pymc-devs/pymc/blob/14d8e9fc03bf9be1c3508b8b4563561480f0b358/pymc/diagnostics.py#L497
        with speed-up modifications from:
        https://jwalton.info/Efficient-effective-sample-size-python/
    """
    M, N = x.shape # (number of chains) X (number of samples)
    
    # calculate autocorrelation
    negative_autocorr = False
    t = 1
    variogram = lambda t: ((x[:, t:] - x[:, :(N - t)])**2).sum() / (M*(N - t))
    rho = np.ones(N)
    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    s2 = rhat(x, param)
    while not negative_autocorr and (t < N):
        rho[t] = 1. - variogram(t)/(2.*s2)
        
        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0
        
        t += 1
        
    S = int(M*N / (1 + 2*rho[1:t].sum()))
    return S

def rank(x):
	"""computes ranks for a 1D array using
	the average method to break ties."""
	r = pd.DataFrame(x).rank().values[:,0] 
	
	if(len(x) != 0): #make sure that x has at least 1 element
		return r
	else:
		raise Exception("You gave a 0 length array.")
		
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
	with open(fname, 'rb') as buff:
		data = pickle.load(buff)  

	model, trace = data['model'], data['trace']
	
	return model, trace
