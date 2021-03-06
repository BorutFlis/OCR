from pdf2image import convert_from_path, convert_from_bytes
import os
import pandas as pd
import numpy as np
import pickle
import glob
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import pytesseract
from docx2pdf import convert
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import string
import scipy.stats as stats
import Tokenizer as tk
import representation as rp
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
import gensim.models
from gensim import utils
import re
import datetime
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from scipy import spatial
from sklearn import metrics
import json


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract22.exe'

class MovingWindow:
    def __init__(self, tokens, window_size, step):
        self.current = -step
        self.last = len(tokens) - window_size + 1
        self.remaining = (len(tokens) - window_size) % step
        self.tokens = tokens
        self.window_size = window_size
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        self.current += self.step
        if self.current < self.last:
            return self.tokens[self.current : self.current + self.window_size]
        elif self.remaining:
            self.remaining = 0
            return self.tokens[-self.window_size:]
        else:
            raise StopIteration


class feature_engineering:
    def add_synonyms(self):
        for k, v in self.feature_dict.items():
            for k2 in v.keys():
                for i in range(len(self.feature_dict[k][k2])):
                    if self.feature_dict[k][k2][i] in self.modelW2V.wv.vocab:
                        most_similar = self.modelW2V.wv.most_similar(self.feature_dict[k][k2][i], topn=3)
                        self.feature_dict[k][k2]=list(set(k[0] for k in most_similar).union(set(self.feature_dict[k][k2])))

    def get_feature_dict(self):
        df = pd.read_csv("NDA Breakdown.csv")
        df = df.loc[~df["Terms"].isna()]
        df.reset_index(inplace=True,drop=True)

        df["Component"].fillna(method="ffill", inplace=True)
        df["Features"].fillna(method="ffill", inplace=True)
        grp = df.groupby(["Component", "Features"]).apply(lambda x: ",".join(x["Terms"]).split(","))
        feature_dict = OrderedDict()
        for tup in grp.index:
            if tup[0] in feature_dict:
                feature_dict[tup[0]][tup[1]] = grp[tup]
            else:
                feature_dict[tup[0]] = OrderedDict({tup[1]: grp[tup]})
        return feature_dict

    def flat_dict(self,dd):
        items = []
        for k, v in dd.items():
            if isinstance(v, dict):
                items.extend(self.flat_dict(v))
            else:
                items.extend(v)
        return items

class model_evaluation:

    def evaluate_basic(self,representation,model):
        preds = model.predict(representation.transform([t[0] for t in self.test_texts]))
        targs = [t[1] for t in self.test_texts]
        print("accuracy: ",metrics.accuracy_score(targs, preds))
        print("precision: ", metrics.precision_score(targs, preds))
        print("recall: ", metrics.recall_score(targs, preds,pos_label=-1))
        print("f1: ", metrics.f1_score(targs, preds))
        print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))

    def evaluate_mulitple(self,representations, models):
        for rep in representations:
            for model in models:
                train=rep.fit_transform(self.texts)
                test=rep.transform([t[0] for t in self.test_texts])
                model.fit(train)
                print(self.evaluate(test,[t[1] for t in self.test_texts],model))


    #the parametres of the functions are the text file the built model and the Type of representation for the text vector
    def evaluate(self,test_texts,results,model,exploratory=False,avgs=None):
        #We intialize the counters for all the measures.
        correct=0
        correct_pos=0
        correct_neg=0
        for i,x in enumerate(test_texts):
            #make prediction
            pred = model.predict(x)
            if pred == results[i]:
                correct += 1
                #we also keep count of correct ones based their class
                if results[i]==1:
                    #sensitivity
                    correct_pos+=1
                elif results[i]==-1:
                    #specificity
                    correct_neg+=1
            #if we set exploratory to True we go through the wrongly classified examples
            elif exploratory==True:
                print(i," ",results[i])
                print(self.test_texts[i])
                print(model.decision_function(x))
                print(model.score_samples(x))
                #if avgs != None:

        len_neg=sum(1 if x==-1 else 0  for x in results)
        len_pos=len(results)-len_neg
        return [correct/len(results),correct_pos/len_pos,correct_neg/len_neg]

class conversion:

    def create_test_set(self):
        # we keep the test examples in separate folder as they are different class
        os.chdir("testset")
        test_txt = []
        for filename in os.listdir():
            # We add image string and the label (files with complete NDAs start with n)
            test_txt.append([pytesseract.image_to_string(Image.open(filename)), 1 if filename[0] == "n" else -1])
        os.chdir("..")
        return test_txt

        # this function converts docx, pdf, jpg to txt


class OcrValidation(conversion,model_evaluation,feature_engineering):


    def __init__(self):
        self.vocabulary={}
        self.train_set=[]
        self.model = None
        self.representation = None
        self.doc2vec = None
        self.exemplar_vec = None
        self.p_value=0.3
        self.date_threshold=datetime.datetime(2018,1,1)
        try:
            self.texts = pickle.load(open("train.p", "rb"))
            self.test_texts= pickle.load(open("test.p","rb"))
        except (OSError, IOError) as e:
            # we call the function that read pictures in tesseract
            exit("File not found: test.p and train.p")
        wnl = WordNetLemmatizer()
        pre_processed = [utils.simple_preprocess(t) for t in self.texts]
        for i in range(len(pre_processed)):
            pre_processed[i] = [wnl.lemmatize(w) for w in pre_processed[i]]
        self.modelW2V = gensim.models.Word2Vec(pre_processed, min_count=5)
        self.feature_dict = self.get_feature_dict()
        # self.add_synonyms()
        # We create the vocabulary which we will use as features in our model
        self.vocabulary = {}
        i = 0
        for k, v in self.feature_dict.items():
            for k2 in v.keys():
                for w in self.feature_dict[k][k2]:
                    if w not in self.vocabulary.keys():
                        self.vocabulary[w] = i
                        i += 1
        sum_rep = rp.SumRepresentation(self.vocabulary, self.feature_dict)
        self.cvec = CountVectorizer(vocabulary=self.vocabulary, tokenizer=tk.LemmaTokenizer())
        self.train_set = sum_rep.fit_transform(self.texts[:75])
        model_sum = OneClassSVM(nu=0.05)
        model_sum.fit(self.train_set)
        self.model = model_sum
        self.representation = sum_rep
        d2v_train = pickle.load(open("doc2vec.p", "rb"))
        d2v = rp.Doc2Vec(d2v_train)
        self.doc2vec = d2v
        self.exemplar_vec = self.doc2vec.model.infer_vector([self.texts[1]])


    def set_date_threshold(self,date):
        m = re.search("(\d{2})(?:-|/)(\d{2})(?:-|/)(\d{4})", date)
        if m != None:
            self.date_threshold = datetime.datetime(int(m.groups()[2]), int(m.groups()[1]), int(m.groups()[0]))
        else:
            print("Invalid format")
        return self.date_threshold

    def make_prediction(self, example_text, model,vectorizer):
        pred_vec=vectorizer.transform([example_text])
        return [model.predict(pred_vec)[0],pred_vec]

    def new_example_json(self,js):
        #js= json.loads(re.sub("\\n"," ",json_text))
        txt = ""
        for page in js["analyzeResult"]["readResults"]:
            for line in page["lines"]:
                txt += line["text"] + "\n"
        return_dict=self.identify_features_window(txt, 5, 10)
        return_dict["probability"]=np.mean([v for k,v in return_dict.items()])
        v_dict=self.check_validity(txt)
        return_dict={**v_dict ,**return_dict}
        return return_dict

    def check_validity(self,text):
        m = re.search("(?<=and) +(\w+) [,\(]? ?hereinafter", text)
        m1 = re.search("(?<=between) +(\w+) [,\(]? ?hereinafter", text)
        return_dict={}
        if m!=None and m1!=None:
            return_dict["Existence of Parties"]= f"YES {m.groups()[0]} and {m1.groups()[0]}"
        else:
            return_dict["Existence of Parties"]= "NO"
        m2 = re.search("(\d{2})(?:-|/)(\d{2})(?:-|/)(\d{4})", text)
        if m2!=None:
            dt = datetime.datetime(int(m2.groups()[2]), int(m2.groups()[1]), int(m2.groups()[0]))
            outdated= "YES" if self.date_threshold>dt else "NO"
            return_dict["Date"]=f"Outdated NDA: {outdated}"
        else:
            return_dict["Date"]="No date of NDA given"
        return return_dict

    def get_document_distance(self,vector):
        #we change the train data to pandas dataframe, due to ease of mean/std calculation
        df=pd.DataFrame(self.train_set.toarray())
        p_values = self.z_score_distribution(vector, df)

        #we nee to initialize a counter to keep track where we are in the double for loop
        i=0
        # we go through each feature and check if any of them are under-represented in the file
        for k ,v in self.feature_dict.items():
            for k2 in v.keys():
                print(f"The feature {k} - {k2}:  {p_values[0][i]}")
                i+=1
        print("Summary:")
        component_df=self.representation.component_values(self.train_set)
        component_vec=self.representation.component_values(vector)
        p_values=self.z_score_distribution(component_vec,component_df)
        for i,k in enumerate(self.feature_dict.keys()):
            print(f"Component {k}: {p_values[0][i]}")

    def z_score_distribution(self,individual,population):
        z_scores = (individual - population.mean().to_numpy()) / population.std().to_numpy()
        return stats.norm.cdf(z_scores)

    def identify_features(self,pdf_path,doc2vec):
        #we download text examples representing features
        features=pickle.load(open("features.p","rb"))
        #we transform all the features to doc2vec embedings
        f_vecs=doc2vec.transform([v for k,v in features.items()])
        paragraphs=[]
        #we go through the pages of the pdf_document
        for page_layout in extract_pages(pdf_path):
            #through all the elements in the page_layout
            for i, element in enumerate(page_layout):
                #we Only append if it is a textCOntainer and it has a certain length
                if isinstance(element, LTTextContainer) and len(element.get_text()) > 20 and re.match('[a-zA-Z]+',element.get_text()[0]) != None:
                    paragraphs.append(element.get_text())
        #we transform all paragraphs in doc2vec embeddings
        paragraphs_d2v=doc2vec.transform(paragraphs)
        #we create in which we will keep which paragraphs are most representative of certain feature. Important it is an ordered dict
        #because the order will used to access the calculated similarities
        features_similarities=OrderedDict({k:[] for k in features.keys()})
        for i,p in enumerate(paragraphs_d2v):
            #we calculate of a paragraph and all the feature examples
            sims=doc2vec.model.wv.cosine_similarities(p.toarray().transpose(),f_vecs.toarray())
            #go through all the features and append their corresponding similarity
            for ix,k in enumerate(features_similarities.keys()):
                features_similarities[k].append([i,sims[ix][ix]])
        #sorting the similarities
        for k in features_similarities.keys():
            features_similarities[k]=sorted(features_similarities[k],key=lambda x: x[1])
        return paragraphs,features_similarities

    def identify_features_window(self,txt,speed=5,window_size=10):
        features={}
        for k,v in self.feature_dict.items():
            features[k]=self.flat_dict(v)
        f_vecs=self.doc2vec.transform([" ".join(v) for k,v in features.items()])
        tokens=utils.simple_preprocess(txt)
        windows=[]
        tokens_iter=MovingWindow(tokens,window_size,speed)
        for ti in tokens_iter:
            window_vec=self.doc2vec.model.infer_vector(ti)
            windows.append(window_vec)
        sum_vec=self.representation.transform([txt])
        component_vec=self.representation.component_values(sum_vec)
        component_df=self.representation.component_values(self.train_set)
        p_values=self.z_score_distribution(component_vec,component_df)
        f_dict_keys=list(self.feature_dict.keys())
        return_dict={}
        for ix,f in enumerate(features.keys()):
            sims = self.doc2vec.model.wv.cosine_similarities(f_vecs[ix].toarray().transpose(), windows)
            sims_flat=[x[i] for i,x in enumerate(sims)]
            most_similar = np.argsort(sims_flat)[-1]

            token_start=most_similar*speed
            token_finish=most_similar*speed+window_size if most_similar<(len(windows)-1) else tokens[-window_size:]
            #print(tokens[token_start:token_finish])
            #print(sims_flat[most_similar])
            #print(p_values[0][f_dict_keys.index(f)])
            return_dict[f]=0.75*sims_flat[most_similar]+0.25*p_values[0][f_dict_keys.index(f)]
        return return_dict


if __name__ == "__main__":
    ocr_inst=OcrValidation()
    #ocr_inst.setup_model()
    #ocr_inst.feature_dict.pop("Sign-off")
    sum_rep=rp.SumRepresentation(ocr_inst.vocabulary,ocr_inst.feature_dict)
    cvec=CountVectorizer(vocabulary=ocr_inst.vocabulary,binary=True)
    d2v_train=pickle.load(open("doc2vec.p","rb"))
    d2v=rp.Doc2Vec(d2v_train)
    ocr_inst.doc2vec=d2v
    ocr_inst.exemplar_vec=ocr_inst.doc2vec.model.infer_vector([ocr_inst.texts[1]])
    model=OneClassSVM(nu=0.05)
    ocr_inst.evaluate_mulitple([cvec,sum_rep,d2v],[model])

    train_set=sum_rep.fit_transform(ocr_inst.texts[:75])
    ocr_inst.train_set=train_set
    model_sum=OneClassSVM(nu=0.05)
    model_sum.fit(train_set)
    ocr_inst.model=model_sum
    ocr_inst.representation=sum_rep




