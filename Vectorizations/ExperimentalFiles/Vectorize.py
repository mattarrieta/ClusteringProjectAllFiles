import os, itertools
import warnings
from matplotlib.pyplot import text
from numpy import vectorize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings
import statistics

warnings.simplefilter(action='ignore', category=FutureWarning)

file_path_l = []

for root, dirs, files in os.walk('InternClustersTextAlexisNoSub'):
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

for i in range(len(text_l)):
    text_l[i].encode("ascii", errors = "ignore").decode()

"""dataset = text_l
List_of_List_of_words = []
for x in range(len(dataset)):
   List_of_List_of_words.append(dataset[x].split())"""

for i in range(len(text_l)):
    if(len(text_l[i]) > 300):
        text_l[i] = text_l[i][0:200]

countvectorizer = CountVectorizer(stop_words= 'english', lowercase = True, max_df = .8, min_df = .005, ngram_range=(1,3))
tfidfvectorizer = TfidfVectorizer(stop_words= 'english', lowercase = True, max_df = .8, min_df = .005, ngram_range=(1,3))

count_wm = countvectorizer.fit_transform(text_l)
tfidf_wm = tfidfvectorizer.fit_transform(text_l)

count_tokens = countvectorizer.get_feature_names_out()
tfidf_tokens = tfidfvectorizer.get_feature_names_out()

df_countvect = pd.DataFrame(data = count_wm.toarray(),index = types_l,columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = types_l,columns = tfidf_tokens)

#print("Count Vectorizer\n")
#print(df_countvect)
#print("\nTD-IDF Vectorizer\n")
#print(df_tfidfvect)

scoreCount = silhouette_score(df_countvect, types_l, metric = "cosine")
scoreTFIDF = silhouette_score(df_tfidfvect, types_l, metric = "cosine")

'''
print("Count Samples")
samplesCount = silhouette_samples(df_countvect, types_l, metric = "cosine")
types = list(set(types_l))
dict = {}
for label in types:
    dict[label] = []

for i in range(len(types_l)):
    dict[types_l[i]].append(samplesCount[i])

for key, value in dict.items():
    print(key, ": ", statistics.mean(value))
print()
print("TFIDF Samples")
samplesTFIDF = silhouette_samples(df_tfidfvect, types_l, metric = "cosine")
types = list(set(types_l))
dict = {}
for label in types:
    dict[label] = []

for i in range(len(types_l)):
    dict[types_l[i]].append(samplesTFIDF[i])

for key, value in dict.items():
    print(key, ": ", statistics.mean(value))
'''
#silhouette_score(pca, training[labels],training_df,'cosine')
print("Score Count:", scoreCount)
print("TFIDF Count:", scoreTFIDF)

pca10 = PCA(435)
print((df_countvect.shape))
scoreCount10 = pca10.fit_transform(df_countvect)
tfidfCount10 = pca10.fit_transform(df_tfidfvect)
print(scoreCount10.shape)
scoreCount10 = silhouette_score(scoreCount10, types_l)
scoreTFIDF10 = silhouette_score(tfidfCount10, types_l)
print("Score Count:", scoreCount10)
print("TFIDF Count:", scoreTFIDF10)
#Look at files
#stop words and stemming
#binary
#how much vocab
#Stemming
#
pca10 = PCA(2)
scoreCount10 = pca10.fit_transform(df_countvect)
tfidfCount10 = pca10.fit_transform(df_tfidfvect)
scoreCount10 = silhouette_score(scoreCount10, types_l)
scoreTFIDF10 = silhouette_score(tfidfCount10, types_l)
print("Score Count:", scoreCount10)
print("TFIDF Count:", scoreTFIDF10)

'''
pca = PCA(2)
df = pca.fit_transform(df_countvect)

kmeans = KMeans(init = "random", n_clusters = 3)
label = kmeans.fit_predict(df)

crosstab = pd.crosstab(types_l, label)
scores = cross_val_score(kmeans, df, types_l, cv=StratifiedKFold(shuffle=True), scoring='accuracy')
cross_val_scores = scores.mean()
print(cross_val_scores)

filtered_label0 = df[label == 0]
filtered_label1 = df[label == 1]
filtered_label2 = df[label == 2]

#Plotting the results
plt.figure()
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = "blue")
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'red')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'black')
plt.show()
plt.figure()
'''