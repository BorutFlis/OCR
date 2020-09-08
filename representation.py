from sklearn.feature_extraction.text import CountVectorizer
import Tokenizer as tk
import pandas as pd

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
        for k in self.feature_dict.keys():
            column_indices=list(map(lambda x: self.cvec.get_feature_names().index(x),self.feature_dict[k]))
            return_vec.append(txt_vec[column_indices].sum())
        return pd.Series(return_vec)

    def fit_transform(self,texts):
        return self.transform(texts)

    def transform(self, texts):
        df = pd.DataFrame(self.cvec.transform(texts).toarray())
        for k in self.feature_dict.keys():
            df[k + "_sum"] = pd.Series()
        df[[k + "_sum" for k in self.feature_dict.keys()]] = df.apply(lambda x: self.sum_vectorize(x), axis=1)
        return df.iloc[:, -12:].to_numpy()
