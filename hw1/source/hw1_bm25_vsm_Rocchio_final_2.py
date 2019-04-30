#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebIR Programming HW1
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import re
import sys
import os
import time
# from itertools import product

# 0. Import file and formulate
if sys.argv[3] == "-i":  
    query_xml = sys.argv[4]
    ranked_list = sys.argv[6]
    model_dir = sys.argv[8]
    NTCIR_dir = sys.argv[10]
else:
    query_xml = sys.argv[3]
    ranked_list = sys.argv[5]
    model_dir = sys.argv[7]
    NTCIR_dir = sys.argv[9]

print('Please wait for a moment...\nThe term-frequency matrix is being built up...')
print('It only take you about 3 minutes.\n')
print('Implement on my computer, pre-processing costs 2\'53.')
print('Implement on my computer, total including documents retrievment costs 2\'55.\n')
# (a) Query processing
start = time.time()
# (b) Vocabulary list
vocab_file = os.path.join(model_dir, 'vocab.all') # 'model/vocab.all'

with open(vocab_file, 'r') as f:
    vocab_list = f.read().splitlines() 
## vocab__list

# (c) File list
file_file = os.path.join(model_dir, 'file-list') # 'model/file-list'

with open(file_file, 'r') as f:
    file_list = f.read().splitlines() 
## file_list
    
# (d) Inverted file
inv_file_file = os.path.join(model_dir, 'inverted-file') # 'model/inverted-file'

with open(inv_file_file,'r') as f:
    inv_f = f.read().splitlines()    
## inv_f[0:10]
 
inv_f_list = [] # index terms co-occur/dependent 
idx = 0
while idx < len(inv_f):
    freq = int(inv_f[idx].split(' ')[-1])
    if freq == 0:
        inv_f_list.append(np.array(list(map(int, inv_f[idx].split(' ')))))
        idx += 1
    else:
        for j in range(freq):
            list_ = [inv_f[idx]] + [inv_f[idx + j + 1]]
            list_ = ' '.join(list_)
            inv_f_list.append(np.array(list(map(int, list_.split(' ')))))
        idx += freq + 1

inv_f_df = np.array(inv_f_list)

del inv_file_file
del inv_f
del inv_f_list

docFreq_vec = np.zeros([len(vocab_list), 1])

vocID_docFreq = np.unique(inv_f_df[:, np.array([0, 1, 2])], axis = 0)
for row in vocID_docFreq:
    vocID, vocID2, docFreq = row
    docFreq_vec[vocID] = docFreq
    if vocID2 != -1:
        docFreq_vec[vocID2] = docFreq
del vocID_docFreq


term_freq_matrix = np.zeros([len(vocab_list), len(file_list)])

vocID_docID_Freq = inv_f_df[:, np.array([0, 1, 3, 4])]

for row in vocID_docID_Freq:
    vocID, vocID2, docID, Freq = row
    
    if vocID2 == -1:
        term_freq_matrix[vocID, docID] += Freq

del vocID_docID_Freq

lenDoc = np.sum(term_freq_matrix, axis = 0)
avg_lenDoc = np.mean(lenDoc)

end = time.time()

print('Here\'s the running time result in this implement.')
print('Pre-processing takes {} seconds.'.format(end - start))

def query2vocIDs(query, weights):
    text = ''.join(list(query * weights))
    voc_ids = [vocab_list.index(w) for w in text]
    return voc_ids

def query_tf(voc_ids):
    q_tf_matrix = np.zeros([len(vocab_list), 1])
    for ID in voc_ids:
        q_tf_matrix[ID] += 1
    return q_tf_matrix[np.unique(voc_ids), :]
    


def bm25_vec_q_docs(k1, k3, b, N, freqs, nums_doc, q_freqs, len_doc, avg_len_doc):
    tf = (k1 + 1) * freqs /  ((k1 * ( 1 - b + b * len_doc / avg_len_doc)) + freqs + 1e-7)
    idf = np.log((N - nums_doc + 0.5) / (nums_doc + 0.5))
    tf_q = (k3 + 1) * q_freqs / (k3 + q_freqs)
    return tf * idf * tf_q

'''
def normalize_fun(col):
    return col / (np.sqrt(np.sum(np.square(col))) + 1e-7)
'''
def calculate_similarity(q_norm, doc_vecs_norm):
    return np.sum(q_norm * doc_vecs_norm, 0)

def extract_doc(cos_sim, threshold):
    num_extracted = np.min([len(np.where(cos_sim >= threshold)[0]), 100])
    extracted_docID = np.argsort(-cos_sim)[:num_extracted]
    extracted_docName = [file_list[ID].lower()[16:] for ID in extracted_docID]
    return extracted_docID, extracted_docName



# 2. Testing - BM25 with Rocchio feedback
test_query_file = query_xml # 'queries/query-test.xml'

# (a) Query processing
tree = ET.parse(test_query_file)
root = tree.getroot()

test_query_data = {}

for i in root[0]:
    test_query_data[i.tag] = []
    
for query in root:
    for item in query:
        test_query_data[item.tag].append(re.sub(r'[\n、。，：;\'"~`$%.?>< 「」]', '', item.text))

test_query_df = pd.DataFrame.from_dict(test_query_data)
## test_query_df.shape # (20, 5)

alpha = 0.1
beta = 0.
gamma = 25

t = 2 if sys.argv[1] == '[-r]' or sys.argv[1] == '[-b]' or sys.argv[1] == '-r' or sys.argv[1] == '-b' else 0
extracted_docs = []

for i in range(len(test_query_df)):
    weights = [50, 0, 0, 2]
    multi_text = test_query_df.iloc[:, 1:].loc[i]
    voc_ids = query2vocIDs(multi_text, weights)
    
    tf = term_freq_matrix[np.unique(np.array(voc_ids)), :]
    df = docFreq_vec[np.unique(np.array(voc_ids))]
    tf_q = query_tf(voc_ids)
    
    doc_vecs = bm25_vec_q_docs(1, 1, 0.75, len(file_list), tf, df, tf_q, lenDoc, avg_lenDoc)

    query_vec = bm25_vec_q_docs(1, 1, 0.75, len(file_list), tf_q, df, tf_q, len(voc_ids), avg_lenDoc)
       
    cos_sim = calculate_similarity(query_vec, doc_vecs)

    if t == 0:
        doc_extracted = extract_doc(cos_sim, 0.)[1]
        extracted_docs.append(doc_extracted)
        
    else:
        for _ in range(t):         
            doc_rel_ID, doc_extracted = extract_doc(cos_sim, 0.)
            doc_irrel_ID = list(set(np.arange(len(file_list))) - set(doc_rel_ID))
            
            vec_rel_docs = np.sum(doc_vecs[:, doc_rel_ID], 1).reshape([-1, 1])
            vec_irrel_docs = np.sum(doc_vecs[:, doc_rel_ID], 1).reshape([-1, 1]) 
            
            new_query_vec = alpha * query_vec + beta * vec_rel_docs / len(doc_rel_ID) - gamma * vec_irrel_docs / len(doc_irrel_ID)
            
            cos_sim = calculate_similarity(new_query_vec, doc_vecs)
            
        extracted_docs.append(doc_extracted)


# list(map(len, extracted_doc))

def save_csv(extracted_doc, file_name):
    with open(file_name, 'w') as f:
        f.write('query_id,retrieved_docs\n')
        for i in range(len(test_query_df)):
            query_id = test_query_df.iloc[i, 0][-3:]
            retrieved_docs = ' '.join(extracted_doc[i])
            f.write('{},{}\n'.format(query_id, retrieved_docs))
            

save_csv(extracted_docs, ranked_list) # 'savings/bm25_vsm_rocchio_final.csv'
# '''0.80216 - Best
# t = 2, weights = [50, 0, 0, 2], 
#alpha = 0.1
#beta = 0.
#gamma = 25

end2 = time.time()

print('Total takes {} seconds.'.format(end2 - start))

print('\nThe file is saved. Please check it. Thanks for your patience.')