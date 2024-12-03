import gradio as gr
import torch
import os
import easyocr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
import argparse

def chatbot_response(user_query):
    """
    Respond to a user query using the ShiftParse Q&A system.
    """
    summary = query_engine.query(user_query)
    return summary

def parse_args():
    """Returns: Command-line arguments"""
    parser = argparse.ArgumentParser(description='AI-driven chatbot interface for document retrieval.')
    parser.add_argument('--data_folder', type=str, default='data/pdf', help='Directory path for document storage')
    parser.add_argument('--language', type=str, default='English', choices=['Romanian', 'Serbian', 'German', 'Hungarian'], help='Language of the use cases')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()

    data_folder = 'data/pdf'

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

    # Create an initial empty "detected_text.txt" file for placeholder
    initial_txt_path = "detected_text.txt"
    with open(initial_txt_path, "w", encoding="utf-8") as f:
        f.write("")  # Create an empty file

    reader = easyocr.Reader(['ro'])
    
    title, description, provider_url, metadata, image, _ = util.fetch_europeana_context(
        api_key=args.key,
        search_query=args.search_query,
        max_result=args.max_results,

        # TODO Optional parameters with defaults
        # data_provider=args.data_provider,
        # institute=args.institute,
        # type=args.type,
        # reusability=args.reusability,
        # country=args.country,
        # language=args.language,
        # start=args.start,
        # sort=args.sort,
        # profile=args.profile,
        # facet=args.facet,
        # qf=args.qf,
        # colour_palette=args.colour_palette,
        # timestamp=args.timestamp,
        # distance=args.distance
    )

    print(f"API Key: {args.key}")
    print(f"Search Query: {args.search_query}")
    print(f"Title: {title}")
    print(f"Description: {description}")
    print(f"Provider URL: {provider_url}")
    print(f"Metadata: {metadata}")

