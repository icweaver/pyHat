import utils
import matplotlib.pyplot as plt

def rankplot(trace, nchains, rows, cols, **kwargs):
    """Returns rank plot for a given chain.
    Parameters
    ----------
    trace : numpy array of trace.
    nchains : number of chains to be plotted; nchains = rows * cols
    rows : number of rows in the plot
    cols: number of cols in the plot
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(17, 10))
    gs = GridSpec(rows, cols)

    r = utils.rank(trace.flatten()).reshape(trace.shape)

    for i in range(nchains):
        ax = fig.add_subplot(gs[i])
        plt.suptitle(kwargs)
        plt.hist(r[i], alpha = 0.8, bins = 50,\
        histtype = 'bar', ec='black', density = True)
        ax.set_title(f"Chain : {i + 1}")

def zscale_hist(trace):
    """Returns histogram of rank normalized chains.
    This is another visual diagnostic tool that can be 
    used to assess mixing and divergence of chains.
    If chains are not well mixed, then the histograms
    will not look like similar Gaussians.
    Parameters
    ----------
    trace : numpy array of trace"""
    
    zscale = utils.zscale(trace)
    
    for chain in zscale:
        plt.hist(chain, alpha = 0.5)

