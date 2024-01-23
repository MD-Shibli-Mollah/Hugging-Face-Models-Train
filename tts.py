#from pytesseract import pytesseract
import pytesseract

#pytesseract.pytesseract.tesseract_cmd = 'C:/OCR/Tesseract-OCR/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd =r'C:/Program Files/Tesseract-OCR//tesseract.exe'
from transformers import pipeline

vqa = pipeline(model="impira/layoutlm-document-qa")
res = vqa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice number?",
)
print(res)