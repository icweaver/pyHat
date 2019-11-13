def rank(x):
	"""computes ranks for a 1D array using
	the average method to break ties."""
	r = pd.DataFrame(x).rank().values[:,0] 
	
	if(len(x) != 0): #make sure that x has at least 1 element
		return r
	else:
		raise Exception("You gave a 0 length array.")