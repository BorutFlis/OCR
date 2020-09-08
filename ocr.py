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

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#class WordRepresentaton:
#    def __init__(self):



class ocr_validation:

    def __init__(self):
        self.vocabulary={}
        self.train_set=[]
        self.model = None
        self.representation = None
        try:
            self.texts = pickle.load(open("train.p", "rb"))
        except (OSError, IOError) as e:
            # we call the function that read pictures in tesseract
            self.texts = self.create_dataset()
            pickle.dump(self.texts, open("train.p","wb"))

    #nu is the probability by which a new example outside the boundaries of the SVM is
    #actually an inlier, higher number means more examples will be classified as outliers
    def setup_model(self,nu=0.05):
        df=pd.read_excel("NDA Breakdown.xlsx")
        #We create the vocabulary which we will use as features in our model
        self.vocabulary={term.lower():i for i,term in enumerate(df["Terms"].dropna().apply(lambda x: x.lower()).unique())}
        #We initialize a word vectorizer we our predefined vocabulary.
        self.cvec = CountVectorizer(vocabulary=self.vocabulary,tokenizer=tk.LemmaTokenizer())
        series_clause = df.iloc[np.where(df["Clause"].notna())[0], 0]
        indices = list(series_clause.index)
        indices.append(len(df))
        self.feature_dict = {series_clause[index]: list(df.iloc[indices[i]:indices[i + 1], 2].dropna()) for i, index in enumerate(series_clause.index)}
        self.feature_dict = {k: list(map(lambda x: x.lower(), v)) for k, v in self.feature_dict.items()}
        self.feature_dict=OrderedDict(sorted(self.feature_dict.items(), key=lambda t: t[0]))
        self.X = self.cvec.fit_transform(self.texts)
        #We initialize a one Class SVM, which is used for anomaly detection
        model = OneClassSVM(gamma='auto',nu=nu)
        #We fit the model on our set
        model.fit(self.X)
        #we return the model that will be able to predict whether the new example adhere to our class
        return model

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


    #the parametres of the functions are the text file the built model and the Type of representation for the text vector
    def evaluate(self,test_texts,results,model,exploratory=False,avgs=None):
        #We intialize the counters for all the measures.
        correct=0
        correct_pos=0
        correct_neg=0
        for i,x in enumerate(test_texts):
            #make prediction
            pred = model.predict([x])
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

    def new_example(self, path):
        example_text = self.doc_ocr_txt(path)
        pred= self.make_prediction(example_text,self.model,self.representation)
        validity="Valid" if pred[0]==1 else "Invalid"
        print(f"The file is: {validity}")
        self.add_to_train_set(example_text)
        self.get_document_distance(pred[1])

    def get_document_distance(self,vector):
        df=pd.DataFrame(self.train_set)
        z_scores = (vector[0] - df.mean().to_numpy()) / (df.std().to_numpy())
        # we go through each feature and check if any of them are under-represented in the file
        for i, k in enumerate(self.feature_dict.keys()):
            p_value = stats.norm.cdf(z_scores[i])
            # this value refers to the probability that we find a lower value in the normal distribution
            if p_value < 0.3:
                # if our p-value is smaller than we print the feature to inform which feature is lacking.
                print(f"The feature {k} is under-represented in the document: {1 - p_value} % of documents score higher")


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
    ocr_inst.feature_dict.pop("Sign-off")
    sum_rep=rp.SumRepresentation(ocr_inst.vocabulary,ocr_inst.feature_dict)
    train_set=sum_rep.fit_transform(ocr_inst.texts[:75])
    ocr_inst.train_set=train_set
    #we load the test data from pickle
    test_txt=pickle.load(open("test.p","rb"))
    test_set= sum_rep.transform(list(x[0] for x in test_txt))
    model_sum=OneClassSVM(nu=0.05)
    model_sum.fit(train_set)
    ocr_inst.model=model_sum
    ocr_inst.representation=sum_rep

