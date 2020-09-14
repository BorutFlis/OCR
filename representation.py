from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import Tokenizer as tk
import pandas as pd
import itertools

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

class SumRepresentation():

    def __init__(self,vocabulary,feature_dict):
        self.cvec=CountVectorizer(vocabulary=vocabulary,tokenizer=tk.LemmaTokenizer())
        self.feature_dict=feature_dict

    def sum_vectorize(self,txt_vec):
        return_vec=[]
        for k,v in self.feature_dict.items():
            for k2 in v.keys():
                column_indices=list(map(lambda x: self.cvec.get_feature_names().index(x),self.feature_dict[k][k2]))
                return_vec.append(txt_vec[column_indices].sum())
        return pd.Series(return_vec)

    def fit_transform(self,texts):
        return self.transform(texts)

    def transform(self, texts):
        df = pd.DataFrame(self.cvec.transform(texts).toarray())
        initial_n_col=len(df.columns)
        for k,v in self.feature_dict.items():
            for k2 in v.keys():
                df[k2+"_sum"] = pd.Series()
        ll=[list(k+"_sum" for k in d.keys()) for d in [self.feature_dict[k] for k in self.feature_dict.keys()]]
        l = list(itertools.chain.from_iterable(ll))
        df[l] = df.apply(lambda x: self.sum_vectorize(x), axis=1)
        new_cols=len(df.columns)-initial_n_col
        return csr_matrix(df.iloc[:, -new_cols:].values)

