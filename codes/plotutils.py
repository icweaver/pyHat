from utils import *
import matplotlib.pyplot as plt

def rankplot(x):
	"""Returns rank plot for a given chain."""
	x_scaled = rank(x)
	plt.hist(x_scaled, alpha = 0.8, bins = 50, histtype = 'bar', ec='black')