import hdbscan
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def clusterHDBSCAN(vectorizedData, labels = None, min_cluster_size = 2 , min_samples = 2, printSummary = False, cluster_selection_method = "eom"):
    """Use HDBSCAN to cluster the vectorized data. Can adjust the HDBSCAN parameters if needed. Returns a data frame with the cluster labels, probabilites and outliers scores.
        If the labels are given the homogenity, completeness, and nmi are also returned."""
    clusterer = hdbscan.HDBSCAN(algorithm = 'best', gen_min_span_tree=True, min_cluster_size = min_cluster_size, min_samples=min_samples, cluster_selection_method=cluster_selection_method)
    clusterer.fit(vectorizedData)

    clusterProbabilities = clusterer.probabilities_
    clusterLabels =   clusterer.labels_
    outlierScores =   clusterer.outlier_scores_
    clusterValidity = clusterer.relative_validity_
    
    #Assign the outliers different categories
    negative = -1
    for x in range(len(clusterLabels)):
        if(clusterLabels[x] == -1):
            clusterLabels[x] = negative
            negative = negative - 1

    #Calculate Clustering metrics
    clusterDict = {"Cluster Label": (clusterLabels), "Cluster probabilities": clusterProbabilities, "Outlier Scores": outlierScores}
    if(labels == None):
        df = pd.DataFrame(data = clusterDict)
        homogenity = None
        complete = None
        nmi = None
    else:
        df = pd.DataFrame(data = clusterDict, index = labels)
        homogenity = homogeneity_score(labels, clusterLabels)
        complete = completeness_score(labels, clusterLabels)
        nmi = normalized_mutual_info_score(labels, clusterLabels)

    #Print metrics if needed
    if(printSummary == True):
        print("Number of clusters:", len(set(clusterLabels)))
        print("Number of documents:", len(clusterLabels))
        if(labels != None):
            print("Homogenity score:", homogenity)
            print("Completeness score:", complete)
            print("Density Based Cluster Validity:", clusterValidity)
            print("Normalized Mutual Info Score:", nmi)
        print(df.to_string())
    
    if(labels == None):
        return df
    else:
        return df, homogenity, complete, nmi

def groupLabels(clusterLabels, text_l):
    """Given a group of labels and texts group the texts based on their labels. Outputs a list of lists of texts."""
    clusterLabelsNumpy = np.array(clusterLabels)
    seen = []
    groups = []
    for i in range(len(set(clusterLabels))):
        groups.append([])
    count = 0
    for x in range(len(clusterLabels)):
        if(clusterLabels[x] not in seen):
            seen.append(clusterLabels[x])
            locations = np.where(clusterLabelsNumpy == clusterLabels[x])[0].tolist()
            for x in (locations):
                groups[count].append(text_l[x])
            count = count + 1
    return groups