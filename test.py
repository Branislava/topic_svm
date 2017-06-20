Ntr = 1000
Ntst = 300

# extracting part of training sample
perm = range(0,len(dataset.train_X))
random.shuffle(perm)
perm = perm[:Ntr]
self.train_X = [dataset.train_X[i] for i in perm]
self.train_Y = [dataset.train_Y[i] for i in perm]
self.gram_train = numpy.zeros((Ntr, Ntr)) 
for i in range(Ntr):
    for j in range(Ntr):
        self.gram_train[i][j] = dataset.gram_train[perm[i]][perm[j]]

# extracting part of test sample
perm2 = range(0,len(dataset.test_X))
random.shuffle(perm2)
perm2 = perm2[:Ntst]
self.test_X = [dataset.test_X[i] for i in perm2]        
self.test_Y = [dataset.test_Y[i] for i in perm2]
self.gram_test = numpy.zeros((Ntst, Ntr)) 
for i in range(Ntst):
    for j in range(Ntr):
        self.gram_test[i][j] = dataset.gram_test[perm2[i]][perm[j]]


# idea: check coolocations
    def amplify_similarities(self, train_set, test_set, train_gram, test_gram):

	# how much to add...
	lambd = 1

	N = len(train_set)
	M = len(test_set)

	# extract coolocations from training set... coolocations within a snippet
	train_snippets_coolocation = dict()
	for i in range(0, N):
	    snippet = train_set[i]
	    for word in snippet:
		# for current word, add current snippet's index
		if word not in train_snippets_coolocation:
		    train_snippets_coolocation[word] = set()
		train_snippets_coolocation[word].add(i)
	print('Training-Coolocations stored')

	# do it for train matrix
	clusters = train_snippets_coolocation.values()
	total = len(clusters)
	j = 0
	for cluster in clusters:
	    # get all 2-sized combinations of a list
	    c = combinations(cluster, 2)
	    for comb in c:
		# symmetrical
		train_gram[comb[0]][comb[1]] += lambd
		train_gram[comb[1]][comb[0]] += lambd
		
	    j += 1         
	    if j % 1000 == 0:
		print('%.2f%%' % (100.0 * j / total))
		
	# diagonal: maximum?
	max_elem = train_gram.max()
	for i in range(0, N):
	    train_gram[i][i] = max_elem + lambd
	print('Training matrix amplified')

	# do it for test matrix
	for i in range(M):
	    snippet = test_set[i]
	    
	    if i % 100 == 0:
		    print(str(i+1) + ' out of ' + str(M))
		    
	    for w in snippet:
		if w in train_snippets_coolocation:
		    cluster = train_snippets_coolocation[w]
		    for j in cluster:
		        test_gram[i][j] += lambd
	print('Test matrix amplified')

	return train_gram, test_gram
