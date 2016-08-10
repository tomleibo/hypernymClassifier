import re
import cPickle as pickle
import numpy as np
from scipy import sparse
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import precision_recall_fscore_support as measure


class HypernymClassifier():

    def __init__(self, filenames, X=None, Y=None):
        self.clf = None
        self.modelFileName = "model"
        self.pairDict = {}
        self.maxPattern = 0
        for file in filenames:
            print ('parsing file: ' + str(file))
            self.maxPattern = self._parseFileIntoMap(file,self.pairDict,self.maxPattern)
        self.pairCount = len(self.pairDict)
        print ('Running on '+ str(self.pairCount) + ' noun pairs')
        print ('Found ' + str(self.maxPattern) + ' different patterns')
        self.X,self.Y = self._convertMapToSparseMatrix(self.pairDict,self.maxPattern+1)
        print ('number of true pairs in Y: ' + str(np.count_nonzero(self.Y)))
        print ('vector length: ' + str (self.maxPattern))
        file = 'x.txt'
        print ('writing X to: '+file)
        self._save_sparse_matrix(file,self.X)
        file = 'y.txt'
        print ('writing Y to: '+file)
        np.savetxt(file,self.Y)

    def _train_and_save(self):
        self.clf = RandomForestClassifier()
        self.clf.fit(self.X, self.Y)
        with open(self.modelFileName, 'w') as f:
            pickle.dump(self.clf, f)

    def _load(self):
        with open(self.modelFileName, 'w') as f:
            self.clf = pickle.load(f)

    def _predict(self,X):
        result = self.clf.predict(X)
        with open("prediction.txt",'w') as f:
            f.write(str(result))

    def _analyze(self):
        train_idx = self.X.shape[0]//4*3
        clf = RandomForestClassifier()
        print('training classifier on '+str(train_idx) + ' vectors')
        clf.fit(self.X[:train_idx], self.Y[:train_idx])
        print('predicting on ' + str(self.X.shape[0] - train_idx) + ' vectors')
        prediction = clf.predict(self.X[train_idx:])
        actual = self.Y[train_idx:]
        tp=[]
        tn=[]
        fn=[]
        fp=[]
        for i in range(len(prediction)):
            if prediction[i] == 1 and actual[i] == 1 and len(tp) < 10:
                tp.append(i+train_idx)
            elif prediction[i] == 0 and actual[i] == 0 and len(tn) < 10:
                tn.append(i+train_idx)
            elif prediction[i] == 0 and actual[i] == 1 and len(fn) < 10:
                fn.append(i+train_idx)
            elif prediction[i] == 1 and actual[i] == 0 and len(fp) < 10:
                fp.append(i+train_idx)
        print ('true positives: '+str(tp))
        print ('false positives: '+str(fp))
        print ('true negatives: '+str(tn))
        print ('false negatives: '+str(fn))


    def _kfold(self, k=10):
        precision = 0
        recall = 0
        f = 0
        for i in range(k):
            print ('Running k-fold number ' + str(i) + '.....')
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(self.X, self.Y, test_size=0.2,
                                                                                 random_state=i)
            metric = self._train_test_and_return_measures(X_train, y_train, X_test, y_test)
            precision += metric[0][0]
            recall += metric[1][0]
            f += metric[2][0]
        print ('final average metrics:')
        print ('precision: ' + str(precision / k))
        print ('recall: ' + str(recall / k))
        print ('f measure: ' + str(f / k))

    def _train_test_and_return_measures(self,X_train,Y_train,X_test,Y_test):
        clf = RandomForestClassifier()
        clf.fit(X_train,Y_train)
        y_pred = clf.predict(X_test)
        return measure(Y_test, y_pred)

    def _readAnnotatedSet(self,filename):
        hypernyms = {}
        with open(filename,'r') as f:
            for line in f.readlines():
                annotated_pair = self._parse_annotated_line(line)
                if annotated_pair[2] == 'True':
                    hypernyms[(annotated_pair[0],annotated_pair[1])] = True
                else:
                    hypernyms[(annotated_pair[0],annotated_pair[1])] = False
        return hypernyms

    def _parse_annotated_line(self,line):
        return re.findall(r"[\S]+",line)



    def _convertMapToSparseMatrix(self,pairDict,vectorLength):
        file = 'input.txt'
        print ('writing pairs and patterns to: '+file)
        hypernym_dict = self._readAnnotatedSet("annotated.txt")
        shape = self._get_intersection_size(self._dict_to_key_set(pairDict),self._dict_to_key_set(hypernym_dict))
        print ('Size of annotated set and shapes of X and Y: ' + str(shape))
        X = sparse.lil_matrix((shape, vectorLength), dtype=np.int8)
        Y = np.zeros((shape))
        i=0
        with open(file,'w') as f:
            for key,val in pairDict.iteritems():
                if key in hypernym_dict:
                    is_hypernym = hypernym_dict.get(key)
                    if is_hypernym:
                        Y[i] = 1
                    else:
                        Y[i] = 0
                    for j in val:
                        X[i,int(j)] = 1
                    f.write(str(i) + '\t' + key[0] + '\t' + key[1] + '\t' + str(val) + '\t' + str(is_hypernym) + '\n' )
                    i+=1
        return X,Y

    def _get_intersection_size(self,set1,set2):
        return len(set1 & set2)

    def _parseFileIntoMap(self,filename,dict,maxPattern):
        with open(filename,'r') as f:
            for line in f:
                path = re.findall(r"[\S]+",line)
                dict,maxPattern = self._insert_path_to_map(path,maxPattern,dict)
        return maxPattern



    def _insert_path_to_map(self, path,maxPattern,dict):
        key = (path[0],path[1])
        entry = dict.get(key)
        if entry is None:
            if len(path) > 2:
                dict[key] = [path[2]]
            else:
                dict[key] = []
        else:
            if len(path) > 2:
                entry.append(path[2])
        if (len(path) > 2 and int(path[2]) > maxPattern):
            maxPattern = int(path[2])
        return dict,maxPattern

    def _dict_to_key_set(self,dict):
        res = set()
        for key in dict.keys():
            res.add(key)
        return res

    def _save_sparse_matrix(self,filename,x):
        with open(filename,'w') as f:
            for i in range(x.shape[0]):
                f.write(str(i) + '\t' + str(x[i]) + '\n')


if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings("ignore")
    clf = HypernymClassifier(["paths-r-00000","wordpairs-r-00000"])
    clf._train_and_save()
    clf._analyze()
    clf._kfold(k=10)
