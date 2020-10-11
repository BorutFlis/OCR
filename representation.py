from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import Tokenizer as tk
import pandas as pd
import itertools
import gensim.models
from gensim import utils

class RepresentationMeta(type):

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'fit_transform') and
                callable(subclass.fit_transform) and
                hasattr(subclass, 'transform') and
                callable(subclass.transform))

class RepresentationInterface(metaclass=RepresentationMeta):
    pass


class Doc2Vec():

    def __init__(self,train_texts):
        pre_processed = [utils.simple_preprocess(t) for t in train_texts]
        self.train_corpus = []
        for i, tokens in enumerate(pre_processed):
            self.train_corpus.append(gensim.models.doc2vec.TaggedDocument(tokens, [i]))
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=5, epochs=20)
        self.model.build_vocab(self.train_corpus)
        self.model.train(self.train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

#    def __repr__(self):
#        print("Doc2Vec Representation")

    def fit_transform(self,texts):
        return self.transform(texts)

    def transform(self, texts):
        vecs=[]
        t_pre_processed = [utils.simple_preprocess(t) for t in texts]
        for t in t_pre_processed:
            vecs.append(self.model.infer_vector(t))
        return csr_matrix(vecs)


class SumRepresentation():

    def __init__(self,vocabulary,feature_dict):
        self.__cvec=CountVectorizer(vocabulary=vocabulary,tokenizer=tk.LemmaTokenizer())
        self.feature_dict=feature_dict

#    def __repr__(self):
#        print("Sum Representation")

    def component_values(self,dataset):
        df=pd.DataFrame(dataset.toarray())
        i=0
        for k,v in self.feature_dict.items():
            df.insert(len(df.columns),k,df.iloc[:,i:i+len(v.keys())].sum(axis=1))
            i+=len(v.keys())
        return df.iloc[:,-len(self.feature_dict.keys()):]

    #function used in dataframe apply it is used for each row
    def sum_vectorize(self,txt_vec):
        return_vec=[]
        #we go through all component keys and through all feature keys to access the vocabulary
        for k,v in self.feature_dict.items():
            for k2 in v.keys():
                #we get the column indexes of the words from the countVectorizer
                column_indices=list(map(lambda x: self.__cvec.get_feature_names().index(x),self.feature_dict[k][k2]))
                return_vec.append(txt_vec[column_indices].sum())
        return pd.Series(return_vec)

    def fit_transform(self,texts):
        return self.transform(texts)

    def transform(self, texts):
        df = pd.DataFrame(self.__cvec.transform(texts).toarray())
        initial_n_col=len(df.columns)
        for k,v in self.feature_dict.items():
            for k2 in v.keys():
                df[k2+"_sum"] = pd.Series()
        ll=[list(k+"_sum" for k in d.keys()) for d in [self.feature_dict[k] for k in self.feature_dict.keys()]]
        l = list(itertools.chain.from_iterable(ll))
        df[l] = df.apply(lambda x: self.sum_vectorize(x), axis=1)
        new_cols=len(df.columns)-initial_n_col
        return csr_matrix(df.iloc[:, -new_cols:].values)

