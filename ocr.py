from pdf2image import convert_from_path, convert_from_bytes
import os
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import pytesseract
from docx2pdf import convert
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
a=convert("input.docx",os.getcwd()+"\\output.pdf")

images = convert_from_path('output.pdf')
images[0].save("out.jpg", "JPEG", quality=80, optimize=True, progressive=True)
print(pytesseract.image_to_string(Image.open('out.jpg')))
os.remove('output.pdf')
