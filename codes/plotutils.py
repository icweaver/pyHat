import utils
import matplotlib.pyplot as plt

def rankplot(trace, nchains, rows, cols, title=None, hist_kwargs=None):
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

    ranks = utils.rank(trace.flatten()).reshape(trace.shape)

    for i, r in enumerate(ranks):
        ax = fig.add_subplot(gs[i])
        if hist_kwargs is None: hist_kwargs = dict()
        ax.hist(r, alpha=0.8, bins=50, histtype="bar", ec="black", 
                **hist_kwargs)
        ax.set_title(f"Chain: {i + 1}")
        fig.suptitle(title, fontsize=16)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #for i in range(nchains):
    #    ax = fig.add_subplot(gs[i])
    #    ax.hist(r[i], alpha = 0.8, bins = 50, histtype = 'bar', ec='black')
    #    ax.set_title(f"Chain : {i + 1}")

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

