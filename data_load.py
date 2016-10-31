import cPickle as pickle
import numpy as np
import os

def load_CIFAR_file(filename):
	'''Load a single file of CIFAR'''
	with open(filename, 'rb') as f:
		datadict= pickle.load(f)
		X = datadict['data']
		Y = datadict['labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float32')
		Y = np.array(Y).astype('int32')
		return X, Y


def load_CIFAR10(directory):
	'''Load all of CIFAR'''
	xs = []
	ys = []
	for k in range(1,6):
		f = os.path.join(directory, "data_batch_%d" % k)
		X, Y = load_CIFAR_file(f)
		xs.append(X)
		ys.append(Y)
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	Xte, Yte = load_CIFAR_file(os.path.join(directory, 'test_batch'))
	return Xtr, Ytr, Xte, Yte





X_train, Y_train, X_test,Y_test=load_CIFAR10('/home/kel/cifar-10-batches-py')

X_train_flat = np.dot(X_train, [0.299, 0.587, 0.114]).reshape(X_train.shape[0],-1).astype(np.float32)
X_train_flat = (X_train_flat/255.0)-0.5
X_test_flat = np.dot(X_test, [0.299, 0.587, 0.114]).reshape(X_test.shape[0],-1).astype(np.float32)
X_test_flat = (X_test_flat/255.0)-.5


