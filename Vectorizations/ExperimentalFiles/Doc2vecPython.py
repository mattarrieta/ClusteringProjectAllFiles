import os, itertools
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import gensim
import gensim.downloader as api
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from matplotlib.pyplot import text
from sklearn.metrics import silhouette_score
from numpy import linalg

file_path_l = []

for root, dirs, files in os.walk('IdealFolders'):
    for filename in files:
        file_path_l.append(os.path.join(root, filename))

text_l = []
for file_path in file_path_l:
    filey = open(file_path,encoding="mbcs")
    text_l.append(filey.read())
    filey.close()
types_l = []
for namey in file_path_l:
    splitted = namey.split('\\')
    types_l.append(splitted[1])
print(types_l)

dataset = api.load('text8')
#dataset = api.load("wiki-english-20171001")
data = [d for d in dataset]

def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
data_for_training = list(tagged_document(data))

model = gensim.models.doc2vec.Doc2Vec(epochs=30)
model.build_vocab(data_for_training)
model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)

dataset = text_l
List_of_List_of_words = []
for x in range(len(dataset)):
   List_of_List_of_words.append(dataset[x].split())

vectors = []
for i in range(len(List_of_List_of_words)):
    vectors.append(model.infer_vector(List_of_List_of_words[i]))
score = silhouette_score(vectors, types_l)
scoreC = silhouette_score(vectors, types_l, metric = "cosine")
scoreM = silhouette_score(vectors, types_l, metric = "manhattan")
print("Full dimensions:", score)
print("Full dimensions:", scoreC)
print("Full dimensinos:", scoreM)