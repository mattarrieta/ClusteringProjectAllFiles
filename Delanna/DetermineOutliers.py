from CleanText import getText
from ClusteringMethods import ClusteringMethods
import numpy as np

class DetermineOutliers:
    def __init__(self, file_path, outlier_method):
        self.file_path = file_path
        self.base_df, self.bounds = self.__get_base_df()
        self.__get_outlier_method(outlier_method)
        self.relevant_data = self.remove_outliers(self.outlier_indices)

    def __get_outlier_method(self, outlier_method):
        if outlier_method == 'dbscan':  # TODO: looks clunky
            self.outlier_indices = self.dbscan()
        elif outlier_method == 'hdbscan':
            self.outlier_indices = self.hdbscan()
        elif outlier_method == 'small-clusters':
            self.outlier_indices = self.small_clusters()
        else:
            raise 'KeyError: Invalid outlier_method. Must be in {dbscan, hdbscan, small-clusters}.'

    def __get_base_df(self):
        vectorizer = AdjustedVectorization(file_path=self.file_path)
        df = vectorizer.tfidf()
        upperbound = min(NumClusters(df).simple, df.shape[0])
        bounds = (NumClusters(df).standard, upperbound)
        print("Num data", len(vectorizer.labels))
        return df, bounds

    def dbscan(self):
        pred_labels = ClusteringMethods(self.base_df, model='dbscan').pred_labels
        outlier_indices = [idx for idx, num in enumerate(pred_labels) if num == -1]
        return outlier_indices

    def hdbscan(self):
        from DataManipulation import DimensionReduction
        reduced_df = DimensionReduction(self.base_df, method='pca-standard', dim=20).red_df

        pred_labels = ClusteringMethods(reduced_df, model='hdbscan').pred_labels
        outlier_indices = [idx for idx, num in enumerate(pred_labels) if num == -1]
        return outlier_indices

    """    def small_clusters(self, min_clust_size=1):
    vectorizer = AdjustedVectorization(file_path=self.file_path)  # vectorize data without outliers
    df = vectorizer.tfidf_weighted(min_df=5, max_df=.50, max_features=self.base_df.shape[0] * 10)
    num_clust = NumClusters(df).silhouette(bounds=self.bounds, print_progress=True, margin=.10)
    pred_labels = ClusteringMethods(df, model='agglomerative', n_clusters=num_clust).pred_labels

    from collections import Counter
    counter, clust_sizes = Counter(pred_labels), range(1, min_clust_size + 1)
    outlier_indices = [idx for idx, pred_label in enumerate(pred_labels) if counter[pred_label] in clust_sizes]
    return outlier_indices"""

    def remove_outliers(self, outlier_indices):
        data, labels = getText(self.file_path)
        for val in reversed(outlier_indices):
            del data[val]
            del labels[val]
        return data, labels

    def reinsert_outliers(self, pred_labels):
        outlier_nums = [num for num in range(-1, -len(self.outlier_indices)-1, -1)]
        for idx, num in zip(self.outlier_indices, outlier_nums):
            pred_labels = np.insert(pred_labels, idx, num)
        return pred_labels











