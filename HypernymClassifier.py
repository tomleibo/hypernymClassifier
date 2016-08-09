import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import tree
import re
from scipy import sparse
import time


class HypernymClassifier():

    def __init__(self, filenames, X=None, Y=None):
        self.clf = svm.SVC(kernel='linear')
        self.pairDict = {}
        self.maxPattern = 0
        for file in filenames:
            self._parseFileIntoMap(file)
        self.pairCount = len(self.pairDict)
        self.X,self.Y = self._convertMapToSparseMatrix(self.pairDict,self.maxPattern+1)
        # print('X: ' + str(self.X.shape[0]))
        # print('Y: '+ str(self.Y.shape[0]))
        print(self.X.todense())



    def _convertMapToSparseMatrix(self,pairDict,vectorLength):
        pairCount = len(pairDict)
        X = sparse.lil_matrix((pairCount, vectorLength), dtype=np.int8)
        Y = np.zeros((pairCount))
        i=0
        for val in pairDict.itervalues():
            if len(val) > 0:
                Y[i] = 1
            for j in val:
                X[i,int(j)] = 1
            i+=1
        return X,Y

    def _parseFileIntoMap(self,filename):
        with open(filename,'r') as f:
            for line in f:
                path = re.findall(r"[\S]+",line)
                self._insert_path_to_map(path)


    def _insert_path_to_map(self, path):
        key = (path[0],path[1])
        entry = self.pairDict.get(key)
        if entry is None:
            if len(path) > 2:
                self.pairDict[key] = [path[2]]
            else:
                self.pairDict[key] = []
        else:
            if len(path) > 2:
                entry.append(path[2])
        if (len(path) > 2 and int(path[2]) > self.maxPattern):
            self.maxPattern = int(path[2])


    def fit(self, X, Y):
        if X is not None and len(X)>0 and Y is not None and len(Y) > 0 :
            self.featureVectorSize = len(X[0])
            self.clf.fit(X, Y)
        else:
            raise "X or Y are empty or None"

    # X: 2d array, containing 1 vector for prediction
    def predict(self, X):
        if X is None or len(X) == 0:
            raise "X is empty"
        # neccassary?
        if len(X[0]) != self.featureVectorSize:
            raise "vector sizes do not match"
        return self.clf.predict(X)

        #for linear kernel SVM or Linear Regression
        # docs: http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#example-svm-plot-separating-hyperplane-py
    def getDecisionLine(self):

        clf=self.clf
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        b = clf.support_vectors_[0]
        yy_down = a * xx + (b[1] - a * b[0])
        b = clf.support_vectors_[-1]
        yy_up = a * xx + (b[1] - a * b[0])

        # plot the line, the points, and the nearest vectors to the plane
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=80, facecolors='none')
        # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

        plt.axis('tight')
        plt.show()
        return a, b

    def getDpMin(self,X,Y,X_test):
        clf = tree.DecisionTreeClassifier(max_depth=1,max_leaf_nodes=2, random_state=0)
        clf.fit(X,Y)
        sample_id=0

        # taken from http://scikit-learn.org/dev/auto_examples/tree/unveil_tree_structure.html
        # and http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

        print('Rules used to predict sample %s: ' % sample_id)

        leave_id = clf.apply(X_test)
        feature = clf.tree_.feature
        for node_id in range(2):
            if leave_id[sample_id] != node_id:
                continue

            if (X_test[sample_id, feature[node_id]] <= clf.tree_.threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("first decision node : (%s %s)"
                  % (threshold_sign,clf.tree_.threshold[node_id]))
            #
            # print("first decision node : (X[%s, %s] (= %s) %s %s)"
            #       % (sample_id,
            #          feature[node_id],
            #          X_test[node_id, feature[node_id]],
            #          threshold_sign,
            #          clf.tree_.threshold[node_id]))


if __name__ == "__main__":

    # clf = HypernymClassifier(["paths-r-00000","wordpairs-r-00000"])
    clf = HypernymClassifier(["data.txt"])


    # X = np.array([[3,4,1],[3,4,0],[1,1,0],[7,2,1],[4,3,1],[6,7,1],[10,10,1],[-1,1,0],[-1,2,0],[-2,-2,0],[-6,-1,0]])
    # print(X.shape)
    '''Y = np.array([1,0,0,1,1,1,1,0,0,0,0])
    X_test = np.array(
        [[0, 0, 0], [-2, -2, 0], [-0, -3, 0], [-5, -1, 0], [1, 1, 1], [2, 3, 0], [2, 5, 0], [8, 8, 0], [7, 9, 0],
         [50, 50, 1]])
    clf.fit(X,Y)
    print clf.predict(X_test)
    clf.getDpMin(X,Y,X_test)
    # print(clf.getDecisionLine())'''