import pandas as pd
import os


class GetTextData:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.file_paths = self.__get_file_paths()
        self.labels, self.data = self.__get_data()

    def __get_file_paths(self):
        text_labels, file_paths = [], []
        for root, dirs, files in os.walk(self.data_folder_path):
            text_labels.append(dirs)
            for file_name in files:
                file_paths.append(os.path.join(root, file_name))
        return file_paths

    def __get_data(self):
        text_labels, text_data = [], []
        for file_path in self.file_paths:
            text_labels.append(file_path.split('\\')[-2])  # folder name
            with open(file_path, encoding="mbcs") as file:
                text_data.append(file.read())
        return text_labels, text_data


class StandardVectorization(GetTextData):
    def __init__(self, data_folder_path):
        GetTextData.__init__(self, data_folder_path)

    def __method_vectorize(self, vectorizer):
        fitted_data = vectorizer.fit_transform(self.data)
        tokens = vectorizer.get_feature_names_out()
        dataframe = pd.DataFrame(data=fitted_data.toarray(), index=self.labels, columns=tokens)
        return dataframe

    def count(self, max_df, min_df, n_grams, binary):
        from sklearn.feature_extraction.text import CountVectorizer
        return self.__method_vectorize(CountVectorizer(stop_words='english'))

    def tfidf(self, min_df=5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        return self.__method_vectorize(TfidfVectorizer(stop_words='english', min_df=min_df))


    def doc2vec(self):
        from gensim.models.doc2vec import Doc2Vec
        try:
            model = Doc2Vec.load(rf'{os.getcwd()}\doc2vec-train-model')
        except FileNotFoundError:
            from TrainFunctions import doc2vec_train
            model = doc2vec_train()
        word_lists = [self.data[i].split() for i in range(len(self.data))]
        vectors = [model.infer_vector(word_lists[i]) for i in range(len(word_lists))]
        return pd.DataFrame(vectors)

    def bert(self, model_type):
        from sentence_transformers import SentenceTransformer
        model_path = rf'{os.getcwd()}\BertModels\{model_type}'
        try:
            return pd.read_csv(model_path)
        except FileNotFoundError:
            model = SentenceTransformer(model_type)
            bert_df = pd.DataFrame(model.encode(self.data, show_progress_bar=True), index=self.labels)
            bert_df.to_csv(model_path, index=False)
            return bert_df


class AdjustedVectorization(GetTextData):
    def __init__(self, data_folder_path, min_freq=0.1):
        GetTextData.__init__(self, data_folder_path)
        self.file_path = data_folder_path
        self.base_freq = min_freq

    def __adjusted_vectorization(self):
        tfidf_df = StandardVectorization(self.file_path).tfidf()
        irrelevant_words = []
        for word in tfidf_df.columns:
            if not tfidf_df[word].ge(self.base_freq).any():
                irrelevant_words.append(word)
        return irrelevant_words

    def count(self):
        irrelevant_words = self.__adjusted_vectorization()
        count_df = StandardVectorization(self.file_path).count()
        return count_df.drop(columns=irrelevant_words)

    def tfidf(self, min_df=5):
        irrelevant_words = self.__adjusted_vectorization()
        method_df = StandardVectorization(self.file_path).tfidf(min_df=min_df)
        return method_df.drop(columns=irrelevant_words)