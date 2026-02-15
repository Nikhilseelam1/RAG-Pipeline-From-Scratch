import fitz


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page_number, page in enumerate(doc):
        text = page.get_text()

        pages_and_texts.append({
            "page_number": page_number,
            "text": text
        })

    return pages_and_texts
