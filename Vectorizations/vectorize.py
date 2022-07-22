import umap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def vectorizeTFIDF(text_l, labels = None, max_df = .8, min_df = .005, ngram_range = (1,2)):
    """Given a list of text returns a data frame with the vectorization of the text. Labels for the text can be included to be the index of the dataframe
        Words with a TFIDF score larger than maxFreq or smaller than minFreq can be removed. How many sequences of characters can be considered with ngramRange."""
    tfidfvectorizer = TfidfVectorizer(stop_words= 'english', lowercase = True, min_df = min_df, max_df = max_df, ngram_range= ngram_range)
    tfidf_wm = tfidfvectorizer.fit_transform(text_l)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    if(labels == None):
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = labels, columns = tfidf_tokens)
    else:
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns = tfidf_tokens)
    return df_tfidfvect

def vectorizeTFIDFUMAP(text_l, labels = None, max_df = .8, min_df = .005, ngram_range = (1,2), n_neighbors = 10, min_dist = 0.1):
    """Given a list of text returns a data frame with the vectorization of the text. Labels for the text can be included to be the index of the dataframe
       Words with a TFIDF score larger than maxFreq or smaller than minFreq can be removed. How many sequences of characters can be considered with ngramRange.
       There is also UMAP dimension reduction done where the parameters n_neighbors and min_dist can be changed."""
    tfidfvectorizer = TfidfVectorizer(stop_words= 'english', lowercase = True, min_df = min_df, max_df = max_df, ngram_range=ngram_range)
    tfidf_wm = tfidfvectorizer.fit_transform(text_l)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    if(labels == None):
            df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = labels, columns = tfidf_tokens)
    else:
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns = tfidf_tokens)
    umapVectorizer = umap.UMAP(metric = 'cosine', n_neighbors = n_neighbors , min_dist = min_dist)
    reducedUMAP = umapVectorizer.fit_transform(df_tfidfvect)
    return reducedUMAP

def vectorizeTFIDFSVD(text_l, labels = None, max_df = .8, min_df = .005, ngram_range = (1,2), n_components = 20):
    """Given a list of text returns a data frame with the vectorization of the text. Labels for the text can be included to be the index of the dataframe
    Words with a TFIDF score larger than maxFreq or smaller than minFreq can be removed. How many sequences of characters can be considered with ngramRange.
    There is also SVD dimension reduction done where the number of components can be chosen."""
    tfidfvectorizer = TfidfVectorizer(stop_words= 'english', lowercase = True, min_df = min_df, max_df = max_df, ngram_range=ngram_range)
    tfidf_wm = tfidfvectorizer.fit_transform(text_l)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    if(labels == None):
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),index = labels, columns = tfidf_tokens)
    else:
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns = tfidf_tokens)
    svd = TruncatedSVD(n_components= n_components)
    reducedSVD = svd.fit_transform(df_tfidfvect)
    return reducedSVD
