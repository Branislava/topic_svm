import os
import numpy
import pickle
import sys
import re
from nltk.corpus import wordnet as wn
import time
import multiprocessing as mp

cache = {}
cache_calls = 0
cache_miss = 0
kernel_calls = 0

total_time = 0
temp_time = 0

class Dataset:
    # class constructor - web snippets dataset
    def __init__(self, raw_train_path, raw_test_path, wiki_topics_path):
        
        # read snippets
        self.train_X, self.train_Y = self.prepare_snippets(raw_train_path)
        self.test_X, self.test_Y = self.prepare_snippets(raw_test_path)
        
        # {'topic:0' : [online, poker, game, ...]}
        self.topics2words = self.read_topics(wiki_topics_path)
        
        # enrich with hidden features
        self.train_X = self.enrich(self.train_X, self.topics2words)
        self.test_X = self.enrich(self.test_X, self.topics2words)
        
        print(self.train_X[0])

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
    def prepare_snippets(self, snippets_path):

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
        
    # helper function: invert dictionary
    def invert(self, my_map):
        inv_map = {}
        for k, v in my_map.iteritems():
            for word in v:
                inv_map[word] = inv_map.get(word, [])
                inv_map[word].append(k)
        
        return inv_map
        
    # helper function: how many times was the word assigned to the certain topic
    def word_in_topic(self, word, topic, mapping):
        # {'online' : [topic:0, topic:13, ...]}
        
        if not word in mapping:
            return 0
        
        return 1.0 * len([w for w in mapping[word] if w == topic])
    
    # helper function: number of words in a snippet assigned to the certain topic
    def number_of_topic_assigments_inna_snippet(self, topic, snippet, words2topics):
        counter = 0
        
        for word in snippet:
            if word in words2topics:
                counter += len(t for t in words2topics[word] if t == topic)
            
        return counter
    
    # helper function: returns list of words ['topic:0', 'topic:0', 'topic:1', ...]
    # depending on the probability
    def discretize_topics(self, topic_vector):
        
        result = list()
        
        # topic_vector:
        # {'topic:0' : 0.3, 'topic:1': 0.12, ...}
        
        for topic in topic_vector:
            p = topic_vector[topic]
            
            # disretize depending on the probability value
            if p == 0:
                to_add = 0
            elif p <= 0.025:
                to_add = 1
            elif p <= 0.05:
                to_add = 2
            elif p <= 0.1:
                to_add = 4
            elif p <= 0.2:
                to_add = 8
            elif p <= 0.4:
                to_add = 16
            elif p <= 0.8:
                to_add = 32
            else:
                to_add = 64
                
            # add word 'topic:i' to_add number of times
            for i in range(to_add):
                result.append(topic)
                
        return result
    
    # enrich with hidden features
    def enrich(self, snippets, topics2words):
        
        # this list contains old features union new features
        new_snippets = list()
        
        alpha = 0.5
        beta = 0.1

        words2topics = self.invert(topics2words)
        
        # for each snippet, topics assigments
        topic_assigments = dict()
        i = 0
        for topic in topics2words:
            topic_assigments[topic] = list()
        
        # list of topics
        topics_list = topics2words.keys()
        
        for snippet in snippets:
            snippet = list(snippet)
            new_words = list()
            for word in snippet:
                topic_vector = dict()
                for topic in topics_list:
                    
                    nk = self.word_in_topic(word, topic, words2topics)
                    n_k = self.word_in_topic(word, topic, self.invert(topic_assigments))
                    nk_tot = 1.0 * len(topics2words[topic])
                    n_k_tot = 1.0 * len(topic_assigments[topic])
                    nm = 1.0 * self.number_of_topic_assigments_inna_snippet(topic, snippet, self.invert(topic_assigments))
                    nm_tot = 1.0 * len(snippet)
                    
                    probability = (nk + n_k + beta)/(nk_tot + n_k_tot + beta) * (nm + alpha)/(nm_tot -1 + alpha)
                    topic_vector[topic] = probability
            new_words = self.discretize_topics(topic_vector)      
            new_snippets.append(snippet + new_words)
        
        return new_snippets

    # read file with hidden topics and store it into dictionary
    def read_topics(self, path):
        i = 0
        topics = dict()
        
        with open(path, 'r') as fin:
            for row in fin:
                # strip newline
                word = row.replace('\n', '').replace('\t', '')
                
                # create new topic
                if i == 0:
                    key = word # remember key
                    
                    if not key in topics:
                        topics[key] = list()
                else:
                    topics[key].append(word)
                    
                i += 1
                # topic read - reset counters
                if i == 201:
                    i = 0
                    
        return topics

    # precomputed training gram matrix
    @staticmethod
    def calculate_gram_matrix_parallel(first, second):
        global calls, miss
        start_time = time.time()
        N = len(first)
        M = len(second)
        same = (N == M)
        gram = numpy.zeros((N, M))
        # leave one core free
        proc_cnt = mp.cpu_count()-1
        print("Executing in parallel on " + str(proc_cnt) + " cores")
        pool = mp.Pool(processes=proc_cnt)
        load_per_core = 1
        load = load_per_core * proc_cnt
        finished = 0
        pairs = []
        indpairs = []
        
        print "Dimensions ", N,"x", M
        pairs = [(first[i],second[j], i, j) for i in range(0,N) for j in range(0,M)]
        n = len(pairs)
        
        results = pool.imap(kartelj_kernel,pairs)
        for i in range(0,n):
            gram[pairs[i][2],pairs[i][3]] = results.next()
            if i%1000 is 0:
                print "%s out of %s calculated. time: %s" % (i, n, time.time()-start_time)
          
        pool.close()
        pool.join()
        
        return gram
    
    # kernel specification
    global kartelj_kernel
    def kartelj_kernel(p):
        # actually, every process makes its own copy of cache and these variables, but still it can have impact
        global use_cache, cache, cache_calls, cache_miss, kernel_calls
        
        x = p[0]
        y = p[1]
        #if i*%100 == 0:
        #    print 'Doing ', id
        kernel_calls+=1
        # there are some short 1-word snippets that weren't recognized by wordnet
        
        if len(x) == 0 or len(y) == 0:
            return 0

        # taking zero synsets in advance, to avoid multiple synsets calls
        xs = set([wn.synsets(t)[0] for t in x])
        ys = set([wn.synsets(t)[0] for t in y])
        
        same =  xs & ys
        
        xs = xs - same
        ys = ys - same
        
        sum_max_pairwise_similarities = len(same)
        pairs_count = len(same)

        while len(xs) > 0 and len(ys) > 0:
            max_similarity = -1
            maxx = None
            maxy = None
            for n1 in xs:
                for n2 in ys:
                    if str(n1) + str(n2) in cache:
                        curr_similarity = cache[str(n1) + str(n2)]
                    elif str(n2) + str(n1) in cache:
                        curr_similarity = cache[str(n2) + str(n1)]
                    else:
                        curr_similarity = n1.path_similarity(n2)
                        if curr_similarity is None:
                            curr_similarity = 0
                        cache[str(n1) + str(n2)] = curr_similarity
                        cache_miss += 1
                    cache_calls += 1
                    if cache_calls%1000000==0 and cache_calls>0:
                        print "Cache miss ratio: ", cache_miss*100.0/cache_calls
                    if curr_similarity > max_similarity:
                        max_similarity = curr_similarity
                        maxx = n1
                        maxy = n2       
            xs.remove(maxx)
            ys.remove(maxy)

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
    if len(sys.argv) < 5:
        print("python dataset.py datasets/train.txt datasets/test.txt datasets/ datasets/wikipedia-topics-classified.txt")
        exit(1)

    raw_train_path = sys.argv[1]
    raw_test_path = sys.argv[2]
    export_dir = sys.argv[3]
    wiki_topics_path = sys.argv[4]

    dset = Dataset(raw_train_path, raw_test_path, wiki_topics_path)
    dataset_path = dset.serialize(export_dir)
    print('Dataset web-snippets exported to ' + dataset_path)
