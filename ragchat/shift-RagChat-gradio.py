import gradio as gr
import torch
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from common_setup import create_common_layout, get_css

import argparse

# Configuration settings
# data_folder = 'pdfs'  # Default folder for document storage
data_folder = 'data/pdf'

# Parsing command line arguments
parser = argparse.ArgumentParser(description='AI-driven chatbot interface for document retrieval.')
parser.add_argument('--data_folder', type=str, default='data/pdf', help='Directory path for document storage')
parser.add_argument('--language', type=str, default='English', choices=['Romanian', 'Serbian', 'German', 'Hungarian'], help='Language of the use cases')
args = parser.parse_args()

llama_id = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = "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"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    llama_id, 
    token=hf_token
    )
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
hf_llm = HuggingFaceLLM(
    model_name=llama_id,
    model_kwargs={
    "token": hf_token, 
    # "device_map": "auto",
    "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
    #"quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.65,
        "top_p": 0.9,
        "top_k": 50,
    },
    tokenizer_name=llama_id,
    tokenizer_kwargs={"token": hf_token, "max_length":4096},
    stopping_ids=stopping_ids
)
embeddings = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # A commonly used embedding model
)
reader = SimpleDirectoryReader(input_dir=args.data_folder, recursive=True)
documents = reader.load_data(num_workers=1)

Settings.embed_model = embeddings
Settings.llm = hf_llm
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def chatbot_response(user_query):
    """
    Respond to a user query using the ShiftParse Q&A system.
    """
    summary = query_engine.query(user_query)
    return summary

css = get_css()

if __name__ == "__main__":
    with gr.Blocks(css=css) as app:
        create_common_layout()  # Apply common layout
        gr.Markdown("# Test-Version ChatBot", elem_id="title")
        gr.Markdown(f"## *Retrieve assets information from path: {args.data_folder}", elem_id="path_info")
        gr.Markdown("## This application is a chatbot that can answer questions using information from documents stored in a specified folder. It uses advanced AI to understand and respond to user queries effectively.", elem_id="description")

        # TAB 1: Chatbot
        with gr.Tab("Chatbot"):
            with gr.Row():
                with gr.Column(scale=2):
                    user_input = gr.Textbox(placeholder="Type your question here...", show_label=False, lines=10)
                with gr.Column(scale=2):
                    chatbot_output = gr.Textbox(placeholder="Curator's Response", show_label=False, lines=10)

            ask_button = gr.Button("Ask", elem_id="button")
            ask_button.click(chatbot_response, inputs=user_input, outputs=chatbot_output)
        
        # TAB 2: Additional Functionality (placeholder for future features)
        with gr.Tab("Additional Features"):
            gr.Markdown("## Coming Soon: More features will be added here.")

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