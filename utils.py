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