import os, itertools
import warnings
from matplotlib.pyplot import text
from numpy import vectorize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class JaccardIndex:
    def __init__(self, dataframe):
        self.df = dataframe
        self.labels = list(self.df.index.values)
        self.clusters = self.separate_cluster_data()
        self.centroids = self.df.groupby(by=self.labels).mean()
        self.cluster_balls = self.find_cluster_balls()
        self.jaccard_idx = self.compute_jaccard_idx()

    def separate_cluster_data(self):
        indices = [self.labels.index(word) for word in set(self.labels)] + [self.df.shape[0]+1]
        indices.sort()
        clusters = [self.df.iloc[val:indices[i+1],:] for i, val in enumerate(indices[:-1])]
        return clusters

    def compute_distances(self, df, point):
        distances = [np.linalg.norm(df.iloc[i, :] - point) for i, val in enumerate(df.index.values)]
        return distances

    def compute_cluster_ball(self, cluster, centroid):
        iqr_vec = 1.5 * (cluster.quantile(0.75) - cluster.quantile(0.25))  # 1.5 * IQR
        max_dist = np.linalg.norm(iqr_vec)
        distances = self.compute_distances(cluster, centroid)
        radius = min(max_dist, max(distances))
        return radius

    def find_cluster_balls(self):
        cluster_balls = []
        for idx, cluster in enumerate(self.clusters):
            centroid = self.centroids.iloc[idx,:]
            radius = self.compute_cluster_ball(cluster, centroid)
            cluster_balls.append((radius, centroid, cluster))
        return cluster_balls

    def compute_jaccard_idx(self):
        jaccard_indices = pd.DataFrame(index=set(self.labels), columns=set(self.labels))
        for (ball1, ball2) in itertools.combinations(self.cluster_balls, 2):
            radius1, centroid1, cluster1 = ball1
            radius2, centroid2, cluster2 = ball2
            union_size = pd.concat([cluster1, cluster2]).shape[0]
            intersection_size = 0
            for i, val in enumerate(cluster1.index.values):
                if np.linalg.norm(cluster1.iloc[i, :] - centroid2) <= radius2:
                    intersection_size += 1
            for i, val in enumerate(cluster2.index.values):
                if np.linalg.norm(cluster2.iloc[i, :] - centroid1) <= radius1:
                    intersection_size += 1
            jaccard_idx = intersection_size / union_size
            jaccard_indices.at[centroid1.name, centroid2.name] = jaccard_idx
            jaccard_indices.at[centroid2.name, centroid1.name] = jaccard_idx
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
        #print(jaccard_indices)
        return jaccard_indices


def main():
    #from DataManipulation import StandardVectorization
    
    file_path_l = []

    for root, dirs, files in os.walk('InternCLustersTextAlexisNoSub'):
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


    countvectorizer = CountVectorizer(stop_words= 'english')
    tfidfvectorizer = TfidfVectorizer(stop_words= 'english')

    count_wm = countvectorizer.fit_transform(text_l)
    tfidf_wm = tfidfvectorizer.fit_transform(text_l)

    count_tokens = countvectorizer.get_feature_names_out()
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()

    df_countvect = pd.DataFrame(data = count_wm.toarray(),index = types_l,columns = count_tokens)
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = types_l,columns = tfidf_tokens)
    JaccardIndex(df_tfidfvect)


if __name__ == '__main__':
    main()