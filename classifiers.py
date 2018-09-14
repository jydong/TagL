# Template by Bruce Maxwell
# Spring 2015
# CS 251 Project 8
#
# Classifier class and child definitions

import sys
import data
import analysis as an
import numpy as np
import math
import scipy.spatial.distance as distance
import os

class Classifier:

    def __init__(self, type):
        '''The parent Classifier class stores only a single field: the type of
        the classifier.  A string makes the most sense.

        '''
        self._type = type

    def type(self, newtype = None):
        '''Set or get the type with this function'''
        if newtype != None:
            self._type = newtype
        return self._type

    def confusion_matrix( self, truecats, classcats ):
        '''Takes in two Nx1 matrices of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.

        '''
        unique_true, mapping_true = np.unique(np.array(truecats.T), return_inverse=True)
        unique_class, mapping_class = np.unique(np.array(classcats.T), return_inverse=True)

        cmtx = np.matrix(np.zeros((len(unique_true), len(unique_true))))

        for i in range(len(mapping_class.tolist())):
            cmtx[mapping_true.tolist()[i], mapping_class.tolist()[i]] += 1

        return cmtx


    def confusion_matrix_str( self, cmtx ):
        '''Takes in a confusion matrix and returns a string suitable for printing.'''
        s = "Confusion Matrix:\n"
        s += "Predicted->"
        for i in range(len(cmtx)):
            s += "%10s" % ("Cluster %d" % (i))
        for i in range(len(cmtx)):
            s += "\n" + 'Cluster %d' % (i)
            for val in cmtx[i].tolist()[0]:
                # print "val", val
                s += "%10d" % (val)
        return s

    def __str__(self):
        '''Converts a classifier object to a string.  Prints out the type.'''
        return str(self._type)



class NaiveBayes(Classifier):
    '''NaiveBayes implements a simple NaiveBayes classifier using a
    Gaussian distribution as the pdf.

    '''

    def __init__(self, dataObj=None, headers=[], categories=None):
        '''Takes in a Data object with N points, a set of F headers, and a
        matrix of categories, one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'Naive Bayes Classifier')
        self.headers = headers # store the headers used for classification
        self.num_classes = 0
        self.num_features = 0 # number of classes and number of features
        self.class_labels = categories # original class labels
        self.class_means = None
        self.class_vars = None
        self.class_scales = None # unique data for the Naive Bayes: means, variances, scales
        if dataObj is not None:
            self.build(dataObj.getdata(self.headers),categories)

    def build( self, A, categories ):
        '''Builds the classifier give the data points in A and the categories'''
        # figure out how many categories there are and get the mapping (np.unique)
        unique, mapping = np.unique(np.array(categories.T), return_inverse=True)
        self.num_classes = len(unique)
        self.num_features = A.shape[1]
        self.class_labels = unique

        self.class_means = np.asmatrix(np.zeros((self.num_classes,self.num_features)))
        self.class_vars = np.asmatrix(np.zeros((self.num_classes, self.num_features)))
        self.class_scales = np.asmatrix(np.zeros((self.num_classes, self.num_features)))

        # compute the means/vars/scales for each class
        for i in range(self.num_classes):
            for j in range(self.num_features):
                self.class_means[i,j] = np.mean(A[(mapping==i),j])
                self.class_vars[i,j] = np.var(A[(mapping==i),j],ddof=1)
                self.class_scales[i,j] = 1./math.sqrt(2 * math.pi * self.class_vars[i,j])


        return

    def classify( self, A, return_likelihoods=False ):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_likelihoods
        is True, it also returns the NxC likelihood matrix.

        '''

        # error check to see if A has the same number of columns as
        # the class means
        if A.shape[1] != self.class_means.shape[1]:
            print "Input A must have the same number of columns as the class means."
            print "A has dimension: ", A.shape
            print "Class mean has dimension: ", self.class_means.shape
            return
        
        # make a matrix that is N x C to store the probability of each
        # class for each data point
        P = np.asmatrix(np.zeros((A.shape[0],self.num_classes)))

        # calculate the probabilities by looping over the classes
        #  with numpy-fu you can do this in one line inside a for loop
        for i in range(self.num_classes):
            for j in range(A.shape[0]):
                a = self.class_vars[i,:]*2
                b = np.square(A[j, :] - self.class_means[i, :])
                P[j,i] = np.prod(np.multiply(self.class_scales[i, :] , np.exp(-b/a)))



        # calculate the most likely class for each data point
        cats = np.argmax(P, axis=1) # take the argmax of P along axis 1

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_likelihoods:
            return cats, labels, P

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nNaive Bayes Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Mean  : ' + str(self.class_means[i,:]) + "\n"
            s += 'Var   : ' + str(self.class_vars[i,:]) + "\n"
            s += 'Scales: ' + str(self.class_scales[i,:]) + "\n"

        s += "\n"
        return s
        
    def write(self, filename):
        '''Writes the Bayes classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the Bayes classifier from the file'''
        # extension
        return

    
class KNN(Classifier):

    def __init__(self, dataObj=None, headers=[], categories=None, K=None,kmeans2=False):
        '''Take in a Data object with N points, a set of F headers, and a
        matrix of categories, with one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'KNN Classifier')
        self.dataObj = dataObj
        self.KNN_headers = headers
        self.num_classes = None
        self.num_features = None
        self.class_means = None
        self.class_labels = categories # original class labels
        self.exemplars = []  # unique data for the KNN classifier: list of exemplars (matrices)
        if dataObj is not None:
            self.build(dataObj.getdata(self.KNN_headers), categories,K,kmeans2)

    def build( self, A, categories, K = None, kmeans2=False ):
        '''Builds the classifier give the data points in A and the categories'''

        # figure out how many categories there are and get the mapping (np.unique)
        unique, mapping = np.unique(np.array(categories.T), return_inverse=True)
        self.num_classes = len(unique)
        self.num_features = A.shape[1]
        self.class_labels = unique

        for i in range(self.num_classes):
            if K is None:
                # append to exemplars a matrix with all of the rows of A where the category/mapping is i
                self.exemplars.append(A[(mapping==i),:])
            else:
                if kmeans2:
                    codebook = an.kmeans2(A[(mapping==i),:], self.KNN_headers, K,whiten=False)[0]
                else:
                    codebook = an.kmeans(A[(mapping == i), :], self.KNN_headers, K, whiten=False)[0]
                self.exemplars.append(codebook)

        return

    def classify(self, A, K=3, return_distances=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matr ix.

        The parameter K specifies how many neighbors to use in the
        distance computation. The default is three.'''

        # error check
        if A.shape[1] != self.exemplars[0].shape[1]:
            print "Input A must have the same number of columns as the exemplars."
            print "A has dimension: ", A.shape
            print "A class of exemplars has dimension: ", self.exemplars[0].shape
            return

        # make a matrix that is N x C to store the distance to each class for each data point
        D = np.asmatrix(np.zeros((A.shape[0],self.num_classes)))

        for i in range(self.num_classes):
            temp = np.asmatrix(np.zeros((A.shape[0],self.exemplars[i].shape[0])))
            for j in range(temp.shape[0]):
                for k in range(temp.shape[1]):
                    temp[j,k] = distance.euclidean(A[j], self.exemplars[i][k])
            temp.sort(axis=1)
            D[:,i] = np.sum(temp[:,:K], axis=1)

        # calculate the most likely class for each data point
        cats = np.argmin(D, axis=1)

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
            s += 'Mean of Exemplars  :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

        s += "\n"
        return s


    def write(self, filename):
        '''Writes the KNN classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the KNN classifier from the file'''
        # extension
        return


# nearest neighbor classifier - consider the closest mean
class NN(Classifier):

    def __init__(self, dataObj=None, headers=[], categories=None, K=None):
        '''Take in a Data object with N points, a set of F headers, and a
        matrix of categories, with one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'KNN Classifier')
        self.dataObj = dataObj
        self.KNN_headers = headers
        self.num_classes = None
        self.num_features = None
        self.class_means = None
        self.class_labels = categories # original class labels
        self.exemplars = []  # unique data for the KNN classifier: list of exemplars (matrices)
        if dataObj is not None:
            self.build(dataObj.getdata(self.KNN_headers), categories,K)

    def build( self, A, categories, K = None ):
        '''Builds the classifier give the data points in A and the categories'''

        # figure out how many categories there are and get the mapping (np.unique)
        unique, mapping = np.unique(np.array(categories.T), return_inverse=True)
        self.num_classes = len(unique)
        self.num_features = A.shape[1]
        self.class_labels = unique

        for i in range(self.num_classes):
            if K is None:
                # append to exemplars a matrix with all of the rows of A where the category/mapping is i
                self.exemplars.append(A[(mapping==i),:])
            else:
                codebook = an.kmeans(A[(mapping==i),:], self.KNN_headers, K,whiten=False)[0]
                self.exemplars.append(codebook)

        return

    def classify(self, A,return_distances=False):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix.

        The parameter K specifies how many neighbors to use in the
        distance computation. The default is three.'''

        # error check
        if A.shape[1] != self.exemplars[0].shape[1]:
            print "Input A must have the same number of columns as the exemplars."
            print "A has dimension: ", A.shape
            print "A class of exemplars has dimension: ", self.exemplars[0].shape
            return

        # make a matrix that is N x C to store the distance to each class for each data point
        D = np.asmatrix(np.zeros((A.shape[0],self.num_classes)))

        for i in range(self.num_classes):
            temp = np.asmatrix(np.zeros((A.shape[0],self.exemplars[i].shape[0])))
            for j in range(temp.shape[0]):
                for k in range(temp.shape[1]):
                    temp[j,k] = distance.euclidean(A[j], self.exemplars[i][k])
            temp.sort(axis=1)
            D[:,i] = np.sum(temp[:,0], axis=1)

        # calculate the most likely class for each data point
        cats = np.argmin(D, axis=1)

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
            s += 'Mean of Exemplars  :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

        s += "\n"
        return s


    def write(self, filename):
        '''Writes the KNN classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the KNN classifier from the file'''
        # extension
        return
    

