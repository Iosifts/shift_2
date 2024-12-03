import os
import gradio as gr
import torch
import easyocr
import fitz  # PyMuPDF
import pdfplumber
from tempfile import NamedTemporaryFile
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# TODO: restate llm modules

def process_img_easyocr(image_files):
    """
        Input: list of images using EasyOCR
    """
    all_texts = []
    for image_bytes in image_files:
        with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_bytes)
            temp_file.seek(0)
            results = reader.readtext(temp_file.name)
        os.remove(temp_file.name)
        all_texts.append(" ".join([result[1] for result in results]))
    return "\n".join(all_texts)

def process_pdf_easyocr(file_content):
    """
        Input: .pdf (of multiple images)
        Uses (cnn + lstm + attention) ocr-model
        Suitable for printed and handwritten text files
    """
    print(__name__, 'easyocr')
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_content)
        temp_pdf.seek(0)
        doc = fitz.open(temp_pdf.name)
        all_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = []
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                image_info = doc.extract_image(xref)
                img_bytes = image_info["image"]
                with NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                    temp_img_file.write(img_bytes)
                    temp_img_file.flush()
                    temp_img_file.close()
                    ocr_result = reader.readtext(temp_img_file.name)
                    text.extend([line[1] for line in ocr_result])
                    os.unlink(temp_img_file.name)
            all_text.append(" ".join(text))
        doc.close()
    return "\n".join(all_text)

def process_pdf_pdfplumber(file_content):
    """
        Input: .pdf (of multiple images)
        Suitable for printed text files (higher char accuracy)
    """
    print(__name__, 'pdfplumber')
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_content)
        temp_pdf.seek(0)
        with pdfplumber.open(temp_pdf.name) as pdf:
            all_text = [page.extract_text() for page in pdf.pages if page.extract_text() is not None]
    return "\n".join(all_text)

def save_text_to_file(text):
    """
    Save detected text to a .txt file and return the path for download.
    """
    with open(initial_txt_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(text)
    return initial_txt_path

# TODO offer inference for trained models

def generate_language_prompt(language):
    prompts = {
        "Romanian": "Summarize the text in Romanian, please! /" + "Rezumatul textului în română!",
        "English": "Summarize the text in English, please!",
        "German": "Summarize the text in German, please! /" + "Zusammenfassung des Textes auf Deutsch!",
        "French": "Summarize the text in French, please! /" + "Résumez le texte en français!",
        "Spanish": "Summarize the text in Spanish, please! /" + "Resumen del texto en español!",
        "Russian": "Summarize the text in Russian, please! /" + "Кратко изложите текст на русском языке!"
    }
    return prompts.get(language, "Summarize the text in English!")  # Default to English if not specified

def translate(input_text, chunk_size=512):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("BlackKakapo/opus-mt-ro-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("BlackKakapo/opus-mt-ro-en")

    # Tokenize the input text into chunks
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Split the input into chunks that do not exceed the max chunk size
    input_chunks = [inputs[:, i:i+chunk_size] for i in range(0, inputs.size(1), chunk_size)]
    
    translated_text = []
    
    # Translate each chunk
    for chunk in input_chunks:
        translated_outputs = model.generate(
            chunk, 
            max_length=chunk_size, 
            num_beams=4,  # For better translation quality
            early_stopping=True
        )

        # Decode and append each chunk's translation
        translated_chunk = tokenizer.decode(translated_outputs[0], skip_special_tokens=True)
        translated_text.append(translated_chunk)

    # Join all the translated chunks together into one string
    return " ".join(translated_text)

def summarize_text(text, language):
    # Use a pre-trained summarization model based on the input language
    if language == 'English':
        model_name = 'facebook/bart-large-cnn'
    elif language == 'Romanian':
        model_name = 'BlackKakapo/opus-mt-ro-en'  # Example translation model
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize input text and summarize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def parse_args():
    """Returns: Command-line arguments"""
    parser = argparse.ArgumentParser('Europeana-api, command-line interface (CLI)')
    parser.add_argument('--path', type=str, help='Path to documents', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # Create an initial empty "detected_text.txt" file for placeholder
    initial_txt_path = "detected_text.txt"
    with open(initial_txt_path, "w", encoding="utf-8") as f:
        f.write("")  # Create an empty file

    reader = easyocr.Reader(['ro'])
    