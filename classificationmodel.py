import os
import numpy
import csv
import pickle

from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import time

from nltk.corpus import wordnet as wn

import multiprocessing as mp

use_cache = False
cache = {}
calls = 0
miss = 0

class ClassificationModel:
    # constructor - for web snippets
    def __init__(self, dataset):
        # classification pipeline
        self.pipeline = make_pipeline(OneVsRestClassifier(SVC(C=1, class_weight='auto', kernel='precomputed')))

        # reading data
        self.train_X, self.train_Y = dataset.train_X, dataset.train_Y
        self.test_X, self.test_Y = dataset.test_X, dataset.test_Y

        # creating Gram-matrices
        self.gram_train = self.calculate_gram_matrix_parallel(self.train_X, self.train_X)
        print("============ Train Gram-Matrix created ============ ")
        self.gram_test = self.calculate_gram_matrix_parallel(self.test_X, self.train_X)
        print("============ Test Gram-Matrix created ============ ")

        # class labels
        self.labels = list(dataset.labels)
        # number of classes
        self.number_of_classes = len(self.labels)
        # building model name (for export)
        self.name = 'words_kartelj_svm'

    global kartelj_kernel

    def kartelj_kernel(p):
        # actually, every process makes its own copy of cache and these variables, but still it can have impact
        global use_cache, cache, calls, miss
        x = p[0]
        y = p[1]
        # there are some short 1-word snippets that weren't recognized by wordnet
        if len(x) == 0 or len(y) == 0:
            return 0

        # if they are the same snippets
        if x == y:
            return 1

        # input: two snippets represented as words list
        # x, y - list of words
        sum_max_pairwise_similarities = 0
        pairs_count = 0

        # taking zero synsets in advance, to avoid multiple synsets calls
        xs = [wn.synsets(t)[0] for t in x]
        ys = [wn.synsets(t)[0] for t in y]

        while len(xs) > 0 and len(ys) > 0:
            max_similarity = 0
            maxi = -1
            maxj = -1
            i = 0
            for n1 in xs:
                j = 0
                for n2 in ys:
                    if use_cache:
                        if str(n1) + str(n2) in cache:
                            curr_similarity = cache[str(n1) + str(n2)]
                        elif str(n2) + str(n1) in cache:
                            curr_similarity = cache[str(n2) + str(n1)]
                        else:
                            curr_similarity = n1.wup_similarity(n2)
                            cache[str(n1) + str(n2)] = curr_similarity
                            miss += 1
                        calls += 1
                        # if calls%100000==0:
                        #    print "Miss ratio: %",round(miss*100/calls), "total calls",calls
                    else:
                        curr_similarity = n1.wup_similarity(n2)
                    if curr_similarity > max_similarity:
                        max_similarity = curr_similarity
                        maxi = i
                        maxj = j
                    j += 1
                i += 1

            # print xs[maxi], ys[maxj], max_similarity
            del xs[maxi]
            del ys[maxj]

            # add them up
            pairs_count += 1
            sum_max_pairwise_similarities += max_similarity

        average = 0
        if pairs_count > 0:
            average = sum_max_pairwise_similarities / pairs_count

        return average

    # precomputed training gram matrix
    def calculate_gram_matrix_parallel(self, first, second):
        start_time = time.time()
        N = len(first)
        M = len(second)
        same = (N == M)
        gram = numpy.zeros((N, M))
        print("Executing in parallel on " + str(mp.cpu_count()) + "cores")
        pool = mp.Pool(processes=mp.cpu_count())
        load_per_core = 25
        load = load_per_core * mp.cpu_count()
        if same:
            tot = N * M / 2
        else:
            tot = N * M
        finished = 0
        pairs = []
        indpairs = []
        for i in range(0, N):
            if same:
                lmt = i
            else:
                lmt = M - 1
            j = 0
            while j <= lmt:
                pairs.append((first[i], second[j]))
                indpairs.append((i, j))
                if len(pairs) == load:
                    results = pool.map(kartelj_kernel, pairs)
                    for k in range(0, len(indpairs)):
                        p = indpairs[k]
                        gram[p[0], p[1]] = results[k]
                        if same:
                            gram[p[1], p[0]] = results[k]
                    finished += len(results)
                    print("Finished " + str(finished) + " out of " + str(tot) + " elapsed time " + str(
                        round(time.time() - start_time)))
                    pairs = []
                    indpairs = []
                j += 1

        pool.close()
        pool.join()

        # for i in range(0, N):
        #     for j in range(0, M):
        #         print str(gram[i][j]) + " ",
        #     print("")

        return gram

    # testing model on test set
    def test(self, filename):
        # predicting labels
        predictions = self.pipeline.predict(self.gram_test)

        # different metrics
        precision, recall, fscore, support = score(self.test_Y, predictions, average=None, labels=self.labels)
        acc = accuracy_score(self.test_Y, predictions)

        # assigning output csv file
        fout = open(filename, "w")
        writer = csv.writer(fout, delimiter=',')
        # writing various statistics
        writer.writerow(['Total questions classified', len(self.test_X)])
        writer.writerow(['Number of classes', self.number_of_classes])

        writer.writerow([])
        writer.writerow(['Class', 'Recall', 'Precision', 'F1 score', 'Support'])
        for i in range(0, self.number_of_classes):
            writer.writerow([self.labels[i], recall[i], precision[i], fscore[i], support[i]])

        writer.writerow([])
        writer.writerow(['Accuracy: ', acc])

        # closing file
        fout.close()

    # train model on training set
    def train(self):
        self.pipeline.fit(self.gram_train, self.train_Y)

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

            self.pipeline.fit(gram_train, train_y)
            pred_y = self.pipeline.predict(gram_test)
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
