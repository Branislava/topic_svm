import requests
import json
import operator
import re

i = 0
topic_id = 0
topics = dict()
with open('wikipedia-topics.txt', 'r') as fin:
    for row in fin:
        # strip newline
        word = row.replace('\n', '').replace('\t', '')
        
        # to classify
        if i == 0:
            key = 'topic:' + str(topic_id)
            topics[key] = list()
        else:
            topics[key].append(word)
            
        i += 1
        # topic read - reset counters
        if i == 201:
            i = 0
            topic_id += 1

for topic in topics:
    # text to be classified
    payload = {'texts': [' '.join(re.escape(w) for w in topics[topic])]}
    headers = {'content-type': 'application/json', 'Authorization': 'Token WmIt3034nAEt'}
    # url request
    url = 'https://api.uclassify.com/v1/uclassify/topics/classify'
    # response object
    r = requests.post(url, data=json.dumps(payload), headers=headers)
    json_response = json.dumps(r.json())
    r = json.loads(json_response)

    max_prob = 0
    max_label = ''
    r = r[0]['classification']
    for d in r:
        
        label = d['className'].upper()
        p = float(d['p'])
        
        if p > max_prob:
            max_prob = p
            max_label = label

    print(max_label)
    print('\n'.join(topics[topic]))