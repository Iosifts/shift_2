import os
import gradio as gr
import torch
import easyocr
import fitz  # PyMuPDF
import pdfplumber
from tempfile import NamedTemporaryFile

# TODO: restate llm modules as needed   
# from transformers import BitsAndBytesConfig
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.embeddings import resolve_embed_model
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.core.embeddings import HuggingFaceEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from common_setup import create_common_layout, get_css

# Create an initial empty "detected_text.txt" file for placeholder
initial_txt_path = "detected_text.txt"
with open(initial_txt_path, "w", encoding="utf-8") as f:
    f.write("")  # Create an empty file

reader = easyocr.Reader(['ro'])

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


def process_handwritten(file_content):
    """
    """
    #Todo
    pass

# TODO offer inference for trained models

# 



# PROMPTING --------------------------------------------------------------------------------------------

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

# GRADIO ----------------------------------------------------------------------------------------------

# def load_text(in_text):
#     # Ensure the directory exists
#     os.makedirs('temp', exist_ok=True)

#     # Save all extracted texts into one large .txt file for retrieval mechanism
#     output_file_path = os.path.join('temp', 'output_text_file.txt')
#     with open(output_file_path, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(in_text))

#     # Load the texts as documents
#     reader = SimpleDirectoryReader(input_dir='temp', recursive=True)
#     documents = reader.load_data(num_workers=1)

#     # Optionally, you can return the documents or content of the file if needed
#     out_text = '\n'.join([doc.get_text() for doc in documents])

#     return out_text

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



css = get_css()

with gr.Blocks(css=css) as app:
    create_common_layout()
    gr.Markdown("# Test-Version Asset2Text",  elem_id="title")
    gr.Markdown('Functionality: text extraction from images/PDFs using OCR, batch processing support; \
                multi-language summarization using AI models; \n File uploads, text detection, \
                and summaries. \nProcess: upload files, detect text, generate summaries in chosen language.', 
                elem_id="description")

    # Tab 1
    with gr.Tab("Tab 1"):
        gr.Textbox("Text Detection and Summarization in different Languages")

    # Tab 2
    with gr.Tab("PDF-based Text Detection and Summarization"):
        with gr.Row():
            with gr.Column(scale=4):
                pdf_input = gr.File(label="Upload PDF File", type="binary", show_label=False)
            with gr.Column(scale=1):    
                language2 = gr.Dropdown(label="Summarization Language", 
                                        choices=["Romanian", "English", "German", 
                                                 "French", "Spanish", "Russian"], 
                                        value="English", show_label=False)
                model1 = gr.Dropdown(label="Detection model", 
                                        choices=["EasyOCR", "TrOCR", "DTrOCR", 
                                                 "Tesseract", "PdfPlumber"],)     
                model2 = gr.Dropdown(label="Summarization model", 
                                        choices=["llama3", "bart"],)                          
                handwritten_checkbox2 = gr.Checkbox(label="Handwritten", value=False, show_label=False)

        with gr.Row():
            with gr.Column():
                pdf_output = gr.Textbox(label="Detected Text (.txt)", lines=10)
            with gr.Column():
                summarization_output2 = gr.Textbox(label="Summarized Text", lines=10)

        with gr.Row():
            with gr.Column():
                process_pdf_button = gr.Button("Detect Text", elem_id="button")
                process_pdf_button.click(
                    lambda file_content, is_handwritten: process_pdf_easyocr(file_content) 
                        if is_handwritten else process_pdf_pdfplumber(file_content),
                    inputs=[pdf_input, handwritten_checkbox2],
                    outputs=pdf_output
                )
            with gr.Column():
                summarize_button = gr.Button("Summarize", elem_id="button")
                summarize_button.click(
                   summarize_text, 
                   inputs=[pdf_output, language2], 
                   outputs=summarization_output2
                )
            
        with gr.Row():
            with gr.Column():
                download_button = gr.Button("Create Text File")
            with gr.Column():
                download_output = gr.File(label="Download Text File")
                download_button.click(
                    lambda text: save_text_to_file(text),
                    inputs=pdf_output,
                    outputs=download_output
                )

    # Tab 3
    with gr.Tab("Image-based Text Detection and Summarization"):
        with gr.Row():
            with gr.Column(scale=4):
                image_input = gr.Files(label="Upload Image Files", type="binary", file_count='multiple', show_label=False)
            with gr.Column(scale=1):
                language1 = gr.Dropdown(label="Summarization Language", 
                                        choices=["Romanian", "English", "German", 
                                                 "French", "Spanish", "Russian"], 
                                        value="English", 
                                        show_label=False)
                #handwritten_checkbox1 = gr.Checkbox(label="Handwritten", value=False, show_label=False)
        with gr.Row():
            with gr.Column():
                image_output = gr.Textbox(label="Detected Text", lines=10)
            with gr.Column():
                summarization_output = gr.Textbox(label="Summarized Text", lines=10)

        with gr.Row():
            with gr.Column():
                process_images_button = gr.Button("Detect Text", elem_id="button")
                process_images_button.click(
                    process_img_easyocr,
                    inputs=[image_input], #, handwritten_checkbox1], 
                    outputs=image_output
                )
            with gr.Column():
                summarize_button = gr.Button("Summarize", elem_id="button")
                summarize_button.click(
                   summarize_text, 
                   inputs=[image_output, language1], 
                   outputs=summarization_output
                )        

    gr.Markdown('''
    ### Shift Text Transformation
    Built upon cutting-edge AI research and technologies including **Llama2**, **Llama3**, and **LlamaIndex**, this platform is designed to facilitate advanced text transformations tailored to diverse audience needs.

    #### Project Collaboration
    ([Schuller's professional page](https://www.professoren.tum.de/schuller-bjoern)) for [SHIFT Europe](https://shift-europe.eu/), aiming to push the boundaries of AI in language processing.

    #### Technologies
    - **Llama2 & Llama3**: Utilizing state-of-the-art language models for nuanced text analysis and generation.
    - **LlamaIndex**: Empowering the backend with efficient indexing to enhance performance and response times.

    #### Learn More
    For more information about the technology and the team, please visit our official website: [Schuller's Lab](http://www.schuller.one/).
    ''')

app.queue()
app.launch(share=True)
