import os
import numpy
import pickle

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from itertools import combinations


class ClassificationModel:
    # constructor - for web snippets
    def __init__(self, dataset, C=1):
        # instances
        self.train_X, self.test_X = dataset.train_X, dataset.test_X
        # class labels
        self.train_Y, self.test_Y = dataset.train_Y, dataset.test_Y
        # gram matrices
        self.gram_train, self.gram_test = dataset.gram_train, dataset.gram_test

        # maybe...
        self.gram_train, self.gram_test = self.amplify_simmilarity(self.train_X, self.test_X, self.gram_train, self.gram_test)

        # class labels
        self.labels = list(dataset.labels)
        # number of classes
        self.number_of_classes = len(self.labels)
        # building model name (for export)
        self.name = 'words_kartelj_svm'
        
        # train model on training set
        self.clf = self.create_train_test(C)

    # idea: check coolocations
    def amplify_simmilarity(self, train_set, test_set, train_gram, test_gram):

        # extract coolocations from training set... coolocations within a snippet
        train_snippets_coolocation = dict()
        for i in range(0, len(train_set)):
            snippet = train_set[i]
            for word in snippet:
                # for current word, add current snippet's index
                if word not in train_snippets_coolocation:
                    train_snippets_coolocation[word] = set()
                train_snippets_coolocation[word].add(i)
        print('Training-Coolocations stored')

        # do it for train matrix
        for cluster in train_snippets_coolocation.values():
            # get all 2-sized combinations of a list
            c = combinations(cluster, 2)
            for comb in c:
                # symmetrical
                train_gram[comb[0]][comb[1]] += 1
                train_gram[comb[1]][comb[0]] += 1
        # diagonal: maximum
        max_elem = train_gram.max()
        for i in range(0, len(train_set)):
            train_gram[i][i] = max_elem + 1
        print('Training matrix amplified')

        # do it for test matrix
        for i in range(0, len(test_set)):
            snippet = test_set[i]
            for word in snippet:
                if word in train_snippets_coolocation:
                    cluster = train_snippets_coolocation[word]
                    for j in cluster:
                        test_gram[i][j] += 1
        print('Test matrix amplified')

        return train_gram, test_gram

    # create, train and test model
    def create_train_test(self, C=1):
        # create
        clf = OneVsRestClassifier(SVC(C=C, class_weight='auto', kernel='precomputed'))        
        
        # train
        clf.fit(self.gram_train, self.train_Y)
        print("Model trained on training set...")
        
        # test
        predictions = clf.predict(self.gram_test)
        print("Model tested on test set...")
        dist = 0.0
        for i in range(0, len(predictions)):
            if predictions[i] == self.test_Y[i]:
                dist += 1.0
        acc = dist / len(predictions)
        print("[C = " + str(C) + "] Accuracy on test set: " + str(acc))
        
        return clf

    # serialize (i.e. write into binary file)
    def serialize(self, directory):
        # create directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = directory + '/' + 'model.' + self.name + '.pkl'
        output = open(filename, 'wb')
        pickle.dump(self, output)

        return filename

    # deserialize (no model instance yet)
    @staticmethod
    def deserialize(path):
        pkl_file = open(path, 'rb')
        model = pickle.load(pkl_file)
        pkl_file.close()

        return model

    # cross-validation method
    def cross_validation(self, folds=5):
        kf = KFold(n_splits=folds, random_state=None, shuffle=True)
        average_accuracy = 0
        k = 1
        for train_index, test_index in kf.split(self.train_X):
            train_y = [self.train_Y[i] for i in train_index]
            gram_train = self.extract_sub_gram(train_index, train_index)

            test_y = [self.train_Y[i] for i in test_index]
            gram_test = self.extract_sub_gram(test_index, train_index)

            self.clf.fit(gram_train, train_y)
            pred_y = self.clf.predict(gram_test)
            average_accuracy += accuracy_score(test_y, pred_y)
            print("=============== " + str(k) + ". fold finished ==============")
            k += 1
        print("Average accuracy in " + str(folds) + " folds: " + str(average_accuracy / folds))

    # helper function - extract certain values from gram matrix
    def extract_sub_gram(self, first, second):
        N = len(first)
        M = len(second)
        gram = numpy.zeros((N, M))

        for i in range(0, N):
            for j in range(0, M):
                gram[i][j] = self.gram_train[first[i]][second[j]]

        return gram

    # tune method
    @staticmethod
    def tune(dataset):
        gram_train, gram_test = dataset.gram_train, dataset.gram_test
        train_Y = dataset.train_Y
        c_range = [0.0001, 0.5, 1, 5, 10, 100, 1000]
        param_grid=dict(estimator__C = c_range)
        svr = OneVsRestClassifier(SVC())
        svr.weight = 'auto'
        svr.kernel = 'precomputed'
        grid = GridSearchCV(svr, param_grid, cv=5, scoring='accuracy')
        grid.fit(gram_train, train_Y)
        print()
        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)
        print("Grid best score:")
        print()
        print (grid.best_score_)
