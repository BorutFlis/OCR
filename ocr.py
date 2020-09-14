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

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ocr_validation:

    def __init__(self):
        self.vocabulary={}
        self.train_set=[]
        self.model = None
        self.representation = None
        self.p_value=0.3
        try:
            self.texts = pickle.load(open("train.p", "rb"))
            self.test_texts= pickle.load(open("test.p","rb"))
        except (OSError, IOError) as e:
            # we call the function that read pictures in tesseract
            self.texts = self.create_dataset()
            pickle.dump(self.texts, open("train.p","wb"))
        wnl = WordNetLemmatizer()
        pre_processed = [utils.simple_preprocess(t) for t in self.texts]
        for i in range(len(pre_processed)):
            pre_processed[i] = [wnl.lemmatize(w) for w in pre_processed[i]]
        self.modelW2V = gensim.models.Word2Vec(pre_processed, min_count=5)


    #nu is the probability by which a new example outside the boundaries of the SVM is
    #actually an inlier, higher number means more examples will be classified as outliers
    def setup_model(self,nu=0.05):
        df=pd.read_csv("newterms.csv")
        self.feature_dict=self.get_feature_dict()
        self.add_synonyms()
        #We create the vocabulary which we will use as features in our model
        self.vocabulary={}
        i=0
        for k,v in self.feature_dict.items():
            for k2 in v.keys():
                for w in self.feature_dict[k][k2]:
                    if w not in self.vocabulary.keys():
                        self.vocabulary[w]=i
                        i+=1
        #We initialize a word vectorizer we our predefined vocabulary.
        self.cvec = CountVectorizer(vocabulary=self.vocabulary,tokenizer=tk.LemmaTokenizer())

        self.X = self.cvec.fit_transform(self.texts)
        #We initialize a one Class SVM, which is used for anomaly detection
        model = OneClassSVM(gamma='auto',nu=nu)
        #We fit the model on our set
        model.fit(self.X)
        #we return the model that will be able to predict whether the new example adhere to our class
        return model

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

    def create_dataset(self):
        os.chdir("dataset")
        txt_data=[]
        for file_name in os.listdir():
            #some file are made of multiple pictures
            if os.path.isdir(file_name):
                os.chdir(file_name)
                txt=""
                for img in os.listdir():
                    txt+=pytesseract.image_to_string(Image.open(img))
                txt_data.append(txt)
                os.chdir("..")
            else:
                txt_data.append(pytesseract.image_to_string(Image.open(file_name)))
        #we return back to the main folder
        os.chdir("..")
        pickle.dump(txt_data, open("train.p","wb"))
        return txt_data

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
                print(results[i])
                print(model.decision_function([x]))
                print(model.score_samples([x]))
                #if avgs != None:

        len_neg=sum(1 if x==-1 else 0  for x in results)
        len_pos=len(results)-len_neg
        return [correct/len(results),correct_pos/len_pos,correct_neg/len_neg]

    def create_test_set(self):
        #we keep the test examples in separate folder as they are different class
        os.chdir("testset")
        test_txt=[]
        for filename in os.listdir():
            #We add image string and the label (files with complete NDAs start with n)
            test_txt.append([pytesseract.image_to_string(Image.open(filename)),1 if filename[0]=="n" else -1])
        os.chdir("..")
        return test_txt

    #this function converts docx, pdf, jpg to txt
    def doc_ocr_txt(self,path):
        ext=path.split(".")[-1]
        if ext=="docx":
            convert(path, os.getcwd() + "\\pred.pdf")
            path = os.getcwd() + "\\pred.pdf"
            ext = path.split(".")[-1]
        if ext == "pdf":
            images = convert_from_path(path)
        elif ext == "jpg":
            images = [Image.open(path)]
        new_txt=""
        for img in images:
            new_txt+=pytesseract.image_to_string(img)
        return new_txt

    def add_to_train_set(self,new_txt):
        #we reset the current working directory as the path of this file
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.texts.append(new_txt)
        pickle.dump(self.texts,open("train.p","wb"))
        return True

    def make_prediction(self, example_text, model,vectorizer):
        pred_vec=vectorizer.transform([example_text])
        return [model.predict(pred_vec)[0],pred_vec]

    def new_example(self, path,add=True):
        example_text = self.doc_ocr_txt(path)
        pred= self.make_prediction(example_text,self.model,self.representation)
        validity="Valid" if pred[0]==1 else "Invalid"
        print(f"The file is: {validity}")
        if add==True:
            self.add_to_train_set(example_text)
        self.get_document_distance(pred[1])

    def get_document_distance(self,vector):
        df=pd.DataFrame(self.train_set.toarray())
        z_scores = (vector.toarray()[0] - df.mean().to_numpy()) / (df.std().to_numpy())

        i=0
        # we go through each feature and check if any of them are under-represented in the file
        for k ,v in self.feature_dict.items():
           for k2 in v.keys():
                p_value = stats.norm.cdf(z_scores[i])
                i+=1
                # this value refers to the probability that we find a lower value in the normal distribution
                if p_value < self.p_value:
                    # if our p-value is smaller than we print the feature to inform which feature is lacking.
                    print(f"The feature {k} - {k2} is under-represented in the document: {1 - p_value} % of documents score higher")


    def convert_to_jpg(self):
        os.chdir("toconvert")
        dataset_folder="\\".join(os.getcwd().split("\\")[:-1])+"\\dataset\\"
        for file_name in os.listdir():
            images= convert_from_path(file_name)
            if len(images)>1:
                folder_name=file_name.strip(".pdf")
                os.mkdir(dataset_folder+folder_name)
                for i,img in enumerate(images):
                    img.save(dataset_folder+folder_name+"\\"+str(i)+".jpg", "JPEG", quality=80, optimize=True, progressive=True)
            else:
                images[0].save(dataset_folder+file_name+".jpg", "JPEG", quality=80, optimize=True, progressive=True)
        os.chdir("..")

    def convert_to_pdf(self):
        os.chdir("docx")
        pdf_folder="\\".join(os.getcwd().split("\\")[:-1])+"\\pdf\\"
        for file_name in os.listdir():
            convert(file_name,pdf_folder+file_name+"docx.pdf")
        os.chdir("..")


if __name__ == "__main__":
    ocr_inst=ocr_validation()
    ocr_inst.setup_model()
    #ocr_inst.feature_dict.pop("Sign-off")
    sum_rep=rp.SumRepresentation(ocr_inst.vocabulary,ocr_inst.feature_dict)
    cvec=CountVectorizer(vocabulary=ocr_inst.vocabulary,binary=True)
    model=OneClassSVM(nu=0.05)
    ocr_inst.evaluate_mulitple([cvec,sum_rep],[model])

    train_set=sum_rep.fit_transform(ocr_inst.texts[:75])
    ocr_inst.train_set=train_set
    model_sum=OneClassSVM(nu=0.05)
    model_sum.fit(train_set)
    ocr_inst.model=model_sum
    ocr_inst.representation=sum_rep



