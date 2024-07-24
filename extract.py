import PyPDF2
from PIL import Image
import os

def extract_text(pdf_path):
    pdf = PyPDF2.PdfFileReader(pdf_path)
    text = ""
    for i in range(0, pdf.getNumPages()):
        page = pdf.pages[i]
        if page.images:
            img = Image.new('RGB', (page.width, page.height))
            img_name = f"{pdf_path[:-4]}_image_{i}.png"
            img.save(f'{os.getcwd()}/{img_name}')  # Saving the image in current directory
        text += page.extractText()
    print(text)
    return text


pdfpath = r"C:\Users\HP\Desktop\SpecializationProject\sample 4.pdf"
extract_text(pdfpath)

