import sys
sys.path.insert(0, r'C:\Users\Matthew Arrieta\Desktop\ClusteringProject\TextInput')
sys.path.insert(0, r'C:\Users\Matthew Arrieta\Desktop\ClusteringProject\TestFiles')
sys.path.insert(0, r'C:\Users\Matthew Arrieta\Desktop\ClusteringProject\Vectorizations')
sys.path.insert(0, r'C:\Users\Matthew Arrieta\Desktop\ClusteringProject\Clustering')
sys.path.insert(0, r'C:\Users\Matthew Arrieta\Desktop\ClusteringProject\KeywordExtraction')

from CleanText import getText
from vectorize import vectorizeTFIDF, vectorizeTFIDFUMAP, vectorizeTFIDFSVD
from ClusterHDBSCAN import clusterHDBSCAN, groupLabels
from LDAKeywordExtraction import keywordResultsClusterLDA

text_l, labels = getText(r"C:\Users\Matthew Arrieta\Desktop\ClusteringProject\TestFiles\KeywordFilesTight", preprocessData = True, RemoveNums = True, lem=True, extendStopWords= False)

vectorized = vectorizeTFIDFUMAP(text_l)

clusterDF = clusterHDBSCAN(vectorized, printSummary= False)

clusterLabels = clusterDF['Cluster Label']

groups = groupLabels(clusterLabels, text_l)

keywordDF = keywordResultsClusterLDA(groups)

print(keywordDF)