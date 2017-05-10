import os
import numpy
import pickle
import sys
import re

import xml.etree.ElementTree as ET
from pandas import DataFrame
from sklearn.utils import shuffle

from nltk.corpus import wordnet as wn

class Dataset:
    # class constructor - web snippets dataset
    def __init__(self, raw_train_path, stop_words_dir, raw_test_path):
        self.train_X, self.train_Y = self.prepare_snippets(raw_train_path, 'training')
        self.test_X, self.test_Y = self.prepare_snippets(raw_test_path, 'test')
        self.samples = list()
        self.number_of_classes = 8
        self.labels = list(set(self.test_Y))

    def prepare_snippets(self, snippets_path, usage='training'):

        data = []
        labels = []

        with open(snippets_path, "r") as fsnippets:
            # for each row (row -- snippet)
            for row in fsnippets:

                row = row.replace("\n", "").decode('utf-8')

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
        print("Usage: python dataset.py ~/PycharmProjects/topic_categorization/dataset/data-web-snippets/train.txt ~/PycharmProjects/topic_categorization/dataset/data-web-snippets/test.txt ~/PycharmProjects/topic_categorization/dataset/")
        exit(1)

    raw_train_path = sys.argv[1]
    raw_test_path = sys.argv[2]
    export_dir = sys.argv[3]

    dset = Dataset(raw_train_path, stop_words_dir, raw_test_path)
    dataset_path = dset.serialize(export_dir)
    print('Dataset web-snippets exported to ' + dataset_path)
