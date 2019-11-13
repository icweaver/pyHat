def rankplot(x):
	"""Returns rank plot for a given chain."""
	x_scaled = r_scale(x)
	plt.hist(x_scaled, alpha = 0.8, bins = 50, histtype = 'bar', ec='black')