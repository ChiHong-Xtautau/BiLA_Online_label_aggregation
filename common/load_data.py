import numpy as np

# Ground truth labels are only used to verify the effectiveness of label aggregation.

def map_classes_to_int(X, classes):
	res = np.ones(X.shape)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if int(X[i][j]) != -1:
				res[i][j] = classes.index(X[i][j])
			else:
				res[i][j] = -1
	return res.astype(np.int64)

def load_all(noisy_labels_url, ground_truth_url):
	X = np.loadtxt(noisy_labels_url)
	Y = np.loadtxt(ground_truth_url)
	classes = []
	for c in Y:
		if c not in classes:
			classes.append(c)
	classes.sort()
	return X, Y, classes

