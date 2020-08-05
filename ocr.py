from pdf2image import convert_from_path, convert_from_bytes
import os
import glob
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import pytesseract
from docx2pdf import convert
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import OneClassSVM

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def setup_model():
    #we call the function that read pictures in tesseract
    texts=create_dataset()
    df=pd.read_excel("NDA Breakdown.xlsx")
    #We create the vocabulary which we will use as features in our model
    vocabulary={term:i for i,term in enumerate(df["Terms"].dropna().unique())}
    #We initialize a word vectorizer we our predefined vocabulary.
    cvec = CountVectorizer(vocabulary=vocabulary)

    X = cvec.fit_transform(texts)
    #We initialize a one Class SVM, which is used for anomaly detection
    model = OneClassSVM(gamma='auto')
    #We fit the model on our set
    model.fit(X)
    #we return the model that will be able to predict whether the new example adhere to our class
    return model

def convert_to_pdf():
    os.chdir("docx")
    pdf_folder="\\".join(os.getcwd().split("\\")[:-1])+"\\pdf\\"
    for file_name in os.listdir():
        convert(file_name,pdf_folder+file_name+"docx.pdf")

def create_dataset():
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
    return txt_data

def convert_to_jpg():
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



