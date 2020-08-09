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


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ocr_validation:

    def __init__(self):
        self.vocabulary={}

    #nu is the probability by which a new example outside the boundaries of the SVM is
    #actually an inlier, higher number means more examples will be classified as outliers
    def setup_model(self,nu=0.05):
        try:
            self.texts = pickle.load(open("train.p", "rb"))
        except (OSError, IOError) as e:
            # we call the function that read pictures in tesseract
            self.texts = self.create_dataset()
            pickle.dump(self.texts,open("train.p"))
        df=pd.read_excel("NDA Breakdown.xlsx")
        #We create the vocabulary which we will use as features in our model
        self.vocabulary={term:i for i,term in enumerate(df["Terms"].dropna().unique())}
        #We initialize a word vectorizer we our predefined vocabulary.
        self.cvec = self.cvec = CountVectorizer(vocabulary=self.vocabulary,binary=True)
        series_clause = df.iloc[np.where(df["Clause"].notna())[0], 0]
        indices = list(series_clause.index)
        indices.append(len(df))
        feature_dict = {series_clause[index]: list(df.iloc[indices[i]:indices[i + 1], 2]) for i, index in enumerate(series_clause.index)}
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
        return txt_data

    #the parametres of the functions are the text file the built model and the Type of representation for the text vector
    def evaluate(self,test_texts,results,model,exploratory=False):
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
                print(x)
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

    def make_prediction(self, path, model):
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
        return model.predict(self.cvec.fit_transform([example_text]))[0]

    def convert_to_jpg(self):
        os.chdir("pdf")
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
    test_txt=pickle.load(open("test.p","rb"))
    test_x= ocr_inst.cvec.transform(list(x[0] for x in test_txt)).toarray()
    print(ocr_inst.evaluate(test_x,list(x[1] for x in test_txt),model))
