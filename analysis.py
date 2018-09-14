# Jingyan Dong
# analysis.py
# CS251
# 2/18/17

import numpy as np
import data
import scipy as sp
import scipy.stats as stats
import math
import scipy.cluster.vq as vq
import random
import scipy.spatial.distance as distance

# Takes in a list of column headers and the Data object and returns a list of
# 2-element lists with the minimum and maximum values for each column.
def data_range(data, headers = []):
	list = []
	if headers == []:
		for header in data.get_headers():
			headers.append(header)
	for header in headers:
		col_index = data.header2matrix[header]
		list.append([data.matrix_data[:, col_index].min(0).tolist()[0][0], data.matrix_data[:, col_index].max(0).tolist()[0][0]])
	return list


# Takes in a list of column headers and the Data object and returns a list of
# the mean values for each column.
def mean(data, headers = []):
	list = []
	if headers == []:
		for header in data.get_headers():
			headers.append(header)
	for header in headers:
		col_index = data.header2matrix[header]
		list.append(data.matrix_data[:, col_index].mean(0).tolist()[0][0])
	return list


# Takes in a list of column headers and the Data object and returns a list of
# the standard deviation for each specified column.
def stdev(data, headers = [] ):
	list = []
	if headers == []:
		for header in data.get_headers():
			headers.append(header)
	for header in headers:
		col_index = data.header2matrix[header]
		list.append(data.matrix_data[:, col_index].std(ddof=1))
	return list

# Takes in a list of column headers and the Data object and returns a matrix with each column normalized
# so its minimum value is mapped to zero and its maximum value is mapped to 1.
def normalize_columns_separately(data, headers = []):
	if headers == []:
		headers =  data.get_headers()
	list = data_range(data, headers)
	m = data.get_data(headers)
	new = m.copy()

	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			new[i, j] = (m[i, j] - list[j][0]) / (list[j][1] - list[j][0])

	return new


# Takes in a list of column headers and the Data object and returns a matrix with each entry normalized so
# that the minimum value (of all the data in this set of columns) is mapped to zero and its maximum value is mapped to 1.
def normalize_columns_together(data, headers = []):
	if headers == []:
		headers =  data.get_headers()

	m = data.get_data(headers)
	min = m.min()
	max = m.max()
	new = m.copy()

	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			new[i, j] = (m[i, j] - min) / (max - min)

	return new

def linear_regression(d, ind, dep):
	y = d.matrix_data[:, d.header2matrix[dep]]
	A = np.empty([d.get_num_rows(), 0])
	for h in ind:
		index = d.header2matrix[h]
		A = np.hstack([A, d.matrix_data[:, index]])
	ones = np.ones([d.get_num_rows(), 1])
	A = np.hstack([A, ones])

	AAinv = np.linalg.inv( np.dot(A.T, A))

	x = np.linalg.lstsq(A, y)
	b = x[0] # solution that provides the best fit regression
	N = len(y)  # sample size
	C = len(b)  # number of coefficients
	df_e = N-C   # SSE df
	df_r = C-1   # SSM df

	error = y - np.dot(A, b)
	sse = np.dot(error.T, error) / df_e
	stderr = np.sqrt(np.diagonal(sse[0, 0] * AAinv))
	t = b.T / stderr
	p = 2*(1 - stats.t.cdf(abs(t), df_e))
	r2 = 1 - error.var() / y.var()

	return b, sse, r2, t, p

# return a list version of linear regression results
def simple_results(result):
	coeff = []
	t = []
	p = []
	for i in range(len(result[0])):
		coeff.append(round(result[0][i, 0], 3))
		t.append(round(result[3][0, i], 3))
		p.append(result[4][0, i])

	return coeff, round(result[1][0,0],3), round(result[2],3), t, p


def test(filename, ind, dep):
	d = data.Data(filename)
	result = linear_regression(d, ind, dep)
	coeff = []
	t = []
	p = []
	for i in range(len(result[0])):
		coeff.append(round(result[0][i,0],3))
		t.append(round(result[3][0,i],3))
		p.append(result[4][0,i])
	print "testing: " + filename
	print "coefficients: ", coeff
	print "sse: ", round(result[1][0,0],3)
	print "R2: ", round(result[2],3)
	print "t: ", t
	print "p: ", p
	print "\n"

def pca(d, headers, normalize = True):
	if normalize:
		A = normalize_columns_separately(d, headers)
	else:
		A = d.get_data(headers)

	m = A.mean(axis=0)
	D = A - m
	U,S,V = np.linalg.svd(D, full_matrices=False)

	# evals = np.square(np.true_divide(S,S.shape[0]-1))
	evals, evecs = np.linalg.eig(np.cov(A,rowvar=False))
	evals[::-1].sort()
	evecs = V.copy()

	pdata = np.transpose(V * np.transpose(D))

	return data.PCAData(headers, pdata, evals, evecs, m)

# Takes in a Data object, a set of headers, and the number of clusters to create
# Computes and returns the codebook, codes, and representation error.
def kmeans_numpy( d, headers, K, whiten = True):
	A = d.get_data(headers)
	W = vq.whiten(A)
	codebook, bookerror = vq.kmeans(W, K)
	codes, error = vq.vq(W, codebook)

	return codebook, codes, error

# The kmeans_init should take in the data, the number of clusters K,
# and an optional set of categories (cluster labels for each data point)
# and return a numpy matrix with K rows, each one repesenting a cluster mean.
# If no categories are given, select the means is to randomly choose K data points
# (rows of the data matrix) to be the first K cluster means
def kmeans_init(d, K, categories = None):
	mean =[]
	if categories is not None: # cluster data by given labels
		for i in range(K): # check groups for each label
			group_size = 0
			group_sum = [0] * d.shape[1]
			for j in range(categories.shape[0]):
				if categories[j,0] == i:
					group_size += 1
					for k in range(d.shape[1]):
						group_sum[k] += d[j,k]
			for s in group_sum:
				mean.append(s/float(group_size))
		return np.matrix(mean).reshape(K, d.shape[1])

	else: # no labels are given
		random_indices = random.sample(range(d.shape[0]),K)
		for ind in random_indices:
			mean.append(np.matrix(d[ind,:]).tolist()[0])
		return np.matrix(mean).reshape(K,d.shape[1])

'''
take in the data and cluster means and return a list or matrix (your choice) of ID values and distances.
The IDs should be the index of the closest cluster mean to the data point.
The default distance metric should be sum-squared distance to the nearest cluster mean.
'''
def kmeans_classify(d, means, metric = "Euclidean"):
	ids = [0] * d.shape[0]
	squared_dis = [float("inf")] * d.shape[0]
	distances = [float("inf")] * d.shape[0]
	for i in range(d.shape[0]):
		for j in range(means.shape[0]):
			if metric == "Euclidean":
				dis = distance.euclidean(d[i], means[j])
				if dis <= distances[i]:
					distances[i] = dis
					ids[i] = j
			elif metric == "L1-Norm":
				dis = distance.cityblock(d[i],means[j])
				if dis <= distances[i]:
					distances[i] = dis
					ids[i] = j
			elif metric == "Hamming":
				dis = distance.hamming(d[i], means[j])
				if dis <= distances[i]:
					distances[i] = dis
					ids[i] = j
			elif metric == "Correlation":
				dis = distance.correlation(d[i], means[j])
				if dis <= distances[i]:
					distances[i] = dis
					ids[i] = j
			elif metric == "Cosine":
				dis = distance.cosine(d[i], means[j])
				if dis <= distances[i]:
					distances[i] = dis
					ids[i] = j

	return np.matrix(ids).reshape(d.shape[0],1), np.matrix(distances).reshape(d.shape[0],1)

def kmeans_algorithm(A, means,metric):
	# set up some useful constants
	MIN_CHANGE = 1e-7
	MAX_ITERATIONS = 100
	D = means.shape[1]
	K = means.shape[0]
	N = A.shape[0]

	# iterate no more than MAX_ITERATIONS
	for i in range(MAX_ITERATIONS):
		# calculate the codes
		codes, errors = kmeans_classify(A, means,metric)

		# calculate the new means
		newmeans = np.zeros_like(means)
		counts = np.zeros((K, 1))
		for j in range(N):
			newmeans[codes[j, 0], :] += A[j, :]
			counts[codes[j, 0], 0] += 1.0

		# finish calculating the means, taking into account possible zero counts
		for j in range(K):
			if counts[j, 0] > 0.0:
				newmeans[j, :] /= counts[j, 0]
			else:
				newmeans[j, :] = A[random.randint(0, A.shape[0]), :]

		# test if the change is small enough
		diff = np.sum(np.square(means - newmeans))
		means = newmeans
		if diff < MIN_CHANGE:
			break

	# call classify with the final means
	codes, errors = kmeans_classify(A, means)

	# return the means, codes, and errors
	return (means, codes, errors)


'''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes and representation errors.
    If given an Nx1 matrix of categories, it uses the category labels
    to calculate the initial cluster means.
'''
def kmeans(d, headers=None, K=None, whiten=True, categories = None,metric="Euclidean"):
	if 'numpy' in str(type(d)):
		A = d
	else:
		A = d.get_data(headers)

	if whiten:
		W = vq.whiten(A)
	else:
		W = A

	codebook = kmeans_init(W, K, categories)
	codebook, codes, errors = kmeans_algorithm(W, codebook,metric)

	# codes = codes.tolist()
	# codes = [c[0] for c in codes]

	return  codebook, codes,  errors

# use vq.kmeans2 for classification
def kmeans2(d,headers=None, K=None, whiten=True):
	if 'numpy' in str(type(d)):
		A = d
	else:
		A = d.get_data(headers)

	if whiten:
		W = vq.whiten(A)
	else:
		W = A

	codebook, bookerror = vq.kmeans2(W, K)
	codes, errors = vq.vq(W, codebook)

	return codebook, codes, errors





if __name__ == '__main__':

	d = np.matrix([[1,2],[3,4],[5,6],[7,8]])
	c = np.matrix([[0],[0],[1],[1]])
	# print c
	# print kmeans_init(d, 2,c)

	# test("data-clean.csv", ["X0", "X1"], "Y")
	# test("data-good.csv", ["X0", "X1"], "Y")
	# test("data-noisy.csv", ["X0", "X1"], "Y")
	# d = data.Data("testdata1.csv")
	# print "----------data_range for header thing1 and thing2--------------"
	# print data_range(d,["thing1", "thing2"])
	# print "\n----------mean for header thing1 and thing2--------------"
	# print mean(d, ["thing1", "thing2"])
	# print "\n----------stdev for header thing1 and thing2--------------"
	# print stdev(d, ["thing1", "thing2"])
	# #print var(d, ["thing1", "thing2"])
	# print "\n-----normalize_columns_separately----: \n", normalize_columns_separately(d, ["thing1", "thing2"])
	# print "\n-----normalize_columns_together:---\n", normalize_columns_together(d,["thing1", "thing2"])


	# d2 = data.Data("colbyhours.csv")
	# d2 = data.Data("testdata2.csv")
	# print "----------data_range--------------"
	# print data_range(d2)
	# print "\n----------mean--------------"
	# print mean(d2)
	# print "\n----------stdev--------------"
	# print stdev(d2)
	#print var(d, ["thing1", "thing2"])
	# print "\n-----normalize_columns_separately----: \n", normalize_columns_separately(d, ["thing1", "thing2"])
	# print "\n-----normalize_columns_together:---\n", normalize_columns_together(d,["thing1", "thing2"])