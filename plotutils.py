from utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def rankplot(trace, nchains, rows, cols):
	"""Returns rank plot for a given chain.
	Parameters
	----------
	trace : numpy array of trace.
	nchains : number of chains to be plotted; nchains = rows * cols
	rows : number of rows in the plot
	cols: number of cols in the plot
	"""
	
	fig = plt.figure(figsize=(17, 10))
	gs = GridSpec(rows, cols)

	for i in range(nchains):
		ax = fig.add_subplot(gs[i])
		trace_scaled = rank(trace[i])
		plt.hist(trace_scaled, alpha = 0.8, bins = 50, histtype = 'bar', ec='black')
		ax.set_title(f"Chain : {i + 1}")