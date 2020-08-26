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
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import  pos_tag
import string
import scipy.stats as stats

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.tokenizer= nltk.RegexpTokenizer(r"\w+")

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(
                w.lower(), t[0].lower()) if t[0].lower() in ["a", "v", "n"] else self.wnl.lemmatize(w.lower()) \
            for w, t in pos_tag(
                self.tokenizer.tokenize(doc)
                )
        ]


class ocr_validation:

    def __init__(self):
        self.vocabulary={}
        try:
            self.texts = pickle.load(open("train.p", "rb"))
        except (OSError, IOError) as e:
            # we call the function that read pictures in tesseract
            self.texts = self.create_dataset()
            pickle.dump(self.texts, open("train.p"))

    #nu is the probability by which a new example outside the boundaries of the SVM is
    #actually an inlier, higher number means more examples will be classified as outliers
    def setup_model(self,nu=0.05):
        df=pd.read_excel("NDA Breakdown.xlsx")
        #We create the vocabulary which we will use as features in our model
        self.vocabulary={term.lower():i for i,term in enumerate(df["Terms"].dropna().apply(lambda x: x.lower()).unique())}
        #We initialize a word vectorizer we our predefined vocabulary.
        self.cvec = CountVectorizer(vocabulary=self.vocabulary,tokenizer=LemmaTokenizer())
        series_clause = df.iloc[np.where(df["Clause"].notna())[0], 0]
        indices = list(series_clause.index)
        indices.append(len(df))
        self.feature_dict = {series_clause[index]: list(df.iloc[indices[i]:indices[i + 1], 2].dropna()) for i, index in enumerate(series_clause.index)}
        self.feature_dict = {k: list(map(lambda x: x.lower(), v)) for k, v in self.feature_dict.items()}
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
        pickle.dump(txt_data, open("train.p"))
        return txt_data

    def add_to_train_set(self,path):
        ext=path.split(".")[-1]
        #we reset the current working directory as the path of this file
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if ext=="docx":
            convert(path, os.getcwd() + "\\pred.pdf")
            path = os.getcwd() + "\\pred.pdf"
            ext = path.split(".")[-1]
        if ext == "pdf":
            images = convert_from_path(path)
        new_txt=""
        for img in images:
            new_txt+=pytesseract.image_to_string(img)
        self.texts.append(new_txt)
        pickle.dump(self.texts,open("train.p"))
        return new_txt


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

    def make_prediction(self, path, model,vectorizer):
        ext=path.split(".")[-1]
        if ext=="docx":
            convert(path, os.getcwd()+"\\pred.pdf")
            path=os.getcwd()+"\\pred.pdf"
            ext = path.split(".")[-1]
        if ext=="pdf":
            images=convert_from_path(path)
        example_text=""
        for img in images:
            example_text+=pytesseract.image_to_string(img)
        pred_vec=vectorizer([example_text])

        return [model.predict(pred_vec)[0],pred_vec]

    def sum_vectorize(self,txt):
        txt_vec = self.cvec.transform(txt).toarray()[0]
        return_vec=[]
        for k in self.feature_dict.keys():
            column_indices=list(map(lambda x: self.cvec.get_feature_names().index(x),self.feature_dict[k]))
            return_vec.append(txt_vec[column_indices].sum())
        return [np.array(return_vec)]


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
    model=ocr_inst.setup_model(nu=0.05)
    pcas={}
    #we transform the test set to a Pandas Dataframe as we will needed soon.
    df_pc=pd.DataFrame(ocr_inst.X.toarray())
    ocr_inst.feature_dict.pop("Sign-off")
    #we load the test data from pickle
    test_txt=pickle.load(open("test.p","rb"))
    test_x= ocr_inst.cvec.transform(list(x[0] for x in test_txt)).toarray()
    test_df=pd.DataFrame(test_x)
    #in this loop we create the reduced dimension with PCAs
    for k in ocr_inst.feature_dict.keys():
        #we get the indices of the words that are part of a specific feature
        column_indices=list(map(lambda x: ocr_inst.cvec.get_feature_names().index(x),ocr_inst.feature_dict[k]))
        pca=PCA(n_components=1)
        df_pc[k+"_pca"]=pca.fit_transform(df_pc.iloc[:,column_indices])
        test_df[k+"_pca"]=pca.transform(test_df.iloc[:,column_indices])
        pcas[k]=pca
    #in this for loop we create attirbutes for the summed features
    for k in ocr_inst.feature_dict.keys():
        #we get the indices of the words that are part of a specific feature
        column_indices = list(map(lambda x: ocr_inst.cvec.get_feature_names().index(x), ocr_inst.feature_dict[k]))
        df_pc[k + "_sum"] = df_pc.iloc[:, column_indices].sum(axis=1)
        test_df[k + "_sum"] = test_df.iloc[:, column_indices].sum(axis=1)
    #we tried the two models
    model_pca=OneClassSVM(nu=0.05)
    model_pca.fit(df_pc.iloc[:75,-24:-12])
    model_sum=OneClassSVM(nu=0.05)
    model_sum.fit(df_pc.iloc[:75,-12:])
    #we evaluate both models
    print(ocr_inst.evaluate(test_df.iloc[:,-24:-12].to_numpy(),list(x[1] for x in test_txt),model_pca))
    print(ocr_inst.evaluate(test_df.iloc[:, -12:].to_numpy(), list(x[1] for x in test_txt), model_sum))
    #we make predictions about the file in the testdata folder with summed features model
    os.chdir("testdata")
    for dc in os.listdir():
        if dc.endswith(".docx"):
            p = ocr_inst.make_prediction(dc, model_sum, ocr_inst.sum_vectorize)
            z_scores = (p[1][0] - df_pc.iloc[:75, -12:].mean().to_numpy()) / (df_pc.iloc[:75, -12:].std().to_numpy())
            validity="Valid" if p[0]==1 else "Invalid" if p[0]==-1 else None
            print(f"Document {dc} is {validity}")
            #we go through each feature and check if any of them are under-represented in the file
            for i, c in enumerate(df_pc.columns[-12:]):
                p_value=stats.norm.cdf(z_scores[i])
                #this value refers to the probability that we find a lower value in the normal distribution
                if p_value<0.3:
                    #if our p-value is smaller than we print the feature to inform which feature is lacking.
                    print(f"The feature {c.strip('_sum')} is under-represented in the document")