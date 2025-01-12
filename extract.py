from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, txt_path):
    reader = PdfReader(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        for page in reader.pages:
            text = page.extract_text()
            if text:
                txt_file.write(text)
                txt_file.write('\n')  # Add a newline after each page

if __name__ == "__main__":
    pdf_path = 'Voyagers.pdf'  # Replace with your PDF file path
    txt_path = 'extracted_text.txt'         # Output text file path
    extract_text_from_pdf(pdf_path, txt_path)
