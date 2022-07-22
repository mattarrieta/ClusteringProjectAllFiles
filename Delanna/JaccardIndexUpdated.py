import numpy as np
from itertools import combinations
import pandas as pd
import os

def summary(df):
    average_scores = df.mean(axis=0, skipna=True)
    variances = df.var(axis=0, skipna=True)
    summary_df = pd.concat([average_scores, variances], axis=1, keys=['Avg Score', 'Variance'])
    return summary_df

class ClustersData:
    def __init__(self, dataframe, labels):
        self.df, self.labels = dataframe, labels
        clusters = [(self.df.loc[self.df.index == label], label) for label in set(self.labels)]
        self.cluster_balls = [self.__get_clusters(cluster, cluster_name) for cluster, cluster_name in clusters]

    @classmethod
    def __get_clusters(cls, cluster, cluster_name):
        iqr_vec = 1.5 * (cluster.quantile(0.75) - cluster.quantile(0.25))  # 1.5 * IQR
        max_dist = np.linalg.norm(iqr_vec)
        centroid = cluster.mean()
        distances = [np.linalg.norm(cluster.iloc[i, :] - centroid) for i, _ in enumerate(cluster.index.values)]
        print("Distances:",distances)
        radius = min(max_dist, max(distances))
        return radius, centroid, cluster, cluster_name


class JaccardIndex(ClustersData):
    def __init__(self, dataframe, labels, save=None):
        print("Jaccrd Initilization")
        self.file_name = save
        ClustersData.__init__(self, dataframe, labels)
        self.score_df = pd.DataFrame(index=set(self.labels), columns=set(self.labels))
        self.__compute_jaccard_idx()

        if self.file_name:
            self.score_df.to_csv(rf'{os.getcwd()}\JaccardIndexValues\{self.file_name}')

    @classmethod
    def __compute_intersection_size(cls, ball1, ball2):  # ball = [radius, centroid, data points, name]
        distances12 = ball1[2].apply(lambda row: np.linalg.norm(row - ball2[1]), axis=1)
        distances21 = ball2[2].apply(lambda row: np.linalg.norm(row - ball1[1]), axis=1)
        intersection12 = distances12[distances12.le(ball2[0])]
        intersection21 = distances21[distances21.le(ball1[0])]
        intersection_size = intersection12.shape[0] + intersection21.shape[0]
        return intersection_size

    def __compute_jaccard_idx(self):
        for ball1, ball2 in combinations(self.cluster_balls, 2):  # ball = [radius, centroid, data points, name]
            union_size = ball1[2].shape[0] + ball2[2].shape[0]
            intersection_size = self.__compute_intersection_size(ball1, ball2)
            jaccard_idx = 1 - intersection_size / union_size

            self.score_df.at[ball1[3], ball2[3]] = jaccard_idx
            self.score_df.at[ball2[3], ball1[3]] = jaccard_idx

    def display(self):
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
        print(self.score_df)