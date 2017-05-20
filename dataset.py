import os
import numpy
import pickle
import sys
import re
import xml.etree.ElementTree as ET
from pandas import DataFrame
from sklearn.utils import shuffle
from nltk.corpus import wordnet as wn
import time
import multiprocessing as mp

use_cache = False
cache = {}
calls = 0
miss = 0

class Dataset:
    # class constructor - web snippets dataset
    def __init__(self, raw_train_path, raw_test_path):
        
        # read snippets
        self.train_X, self.train_Y = self.prepare_snippets(raw_train_path, 'training')
        self.test_X, self.test_Y = self.prepare_snippets(raw_test_path, 'test')

        # creating Gram-matrices
        self.gram_train = Dataset.calculate_gram_matrix_parallel(self.train_X, self.train_X)
        print("============ Train Gram-Matrix created ============")
        self.gram_test = Dataset.calculate_gram_matrix_parallel(self.test_X, self.train_X)
        print("============ Test Gram-Matrix created ============")
        
        # some app stuff
        self.samples = list()
        self.number_of_classes = 8
        self.labels = list(set(self.test_Y))

    # read snippets
    def prepare_snippets(self, snippets_path, usage='training'):

        data = []
        labels = []

        with open(snippets_path, "r") as fsnippets:
            # for each row (row -- snippet)
            for row in fsnippets:

                # removing newlines if any
                row = row.replace("\n", "")
                row = re.sub(r'[^\x00-\x7f]',r'', row)

                if len(row) < 3:
                    continue

                # removing xml tags
                row = re.sub(r'<[^>]*>', '', row)
                # removing numbers
                row = re.sub(r'\d+', '', row)
                # removing punctuation
                # row = re.sub(r'[^\w\s]', '', row)
                # removing multiple spaces
                row = re.sub(r' +', ' ', row)
                # splitting words from row
                terms_with_duplicates = row.split(" ")

                # last word represents class label
                number_of_words = len(terms_with_duplicates)
                label = terms_with_duplicates[number_of_words - 1]

                # removing label from feature terms
                terms_with_duplicates.remove(label)
                # eliminating duplicates
                terms = set(terms_with_duplicates)

                # for each snippet
                final_words_set = set()
                for word in terms:
                    # adding original word
                    # add 1st, or any, because later it won't matter!
                    synsets = wn.synsets(word)
                    if len(synsets) > 0:
                        syn = synsets[0]
                        final_words_set.add(syn.lemmas()[0].name().encode('utf-8'))

                data.append(list(final_words_set))
                labels.append(label)

            return data, labels

    # precomputed training gram matrix
    @staticmethod
    def calculate_gram_matrix_parallel(first, second):
        start_time = time.time()
        N = len(first)
        M = len(second)
        same = (N == M)
        gram = numpy.zeros((N, M))
        print("Executing in parallel on " + str(mp.cpu_count()) + " cores")
        pool = mp.Pool(processes=mp.cpu_count())
        load_per_core = 1
        load = load_per_core * mp.cpu_count()
        if same:
            tot = N * (M + 1) / 2
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
                if len(pairs) == 1:
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
        
        return gram
    
    # kernel specification
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
                            curr_similarity = n1.path_similarity(n2)
                            cache[str(n1) + str(n2)] = curr_similarity
                            miss += 1
                        calls += 1
                        if calls%100000==0:
                            print("Miss ratio: " + str(round(miss*100/calls)) + " total calls " + str(calls))
                    else:
                        curr_similarity = n1.path_similarity(n2)
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

    # serialize (i.e. write into binary file)
    def serialize(self, directory):
        # create directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = directory + '/' + 'dataset.' + 'classes_' + str(self.number_of_classes) + '.pkl'
        output = open(filename, 'wb')
        pickle.dump(self, output)

        return filename

    # deserialize
    @staticmethod
    def deserialize(path):
        pkl_file = open(path, 'rb')
        dataset = pickle.load(pkl_file)
        pkl_file.close()

        return dataset
    
# main function
if __name__ == "__main__":

    # checking number of args
    if len(sys.argv) < 4:
        print("Usage: python dataset.py ~/PycharmProjects/topic_categorization/bin/datasets/train.txt ~/PycharmProjects/topic_categorization/bin/datasets/test.txt ~/PycharmProjects/topic_categorization/bin/datasets/")
        exit(1)

    raw_train_path = sys.argv[1]
    raw_test_path = sys.argv[2]
    export_dir = sys.argv[3]

    dset = Dataset(raw_train_path, raw_test_path)
    dataset_path = dset.serialize(export_dir)
    print('Dataset web-snippets exported to ' + dataset_path)
