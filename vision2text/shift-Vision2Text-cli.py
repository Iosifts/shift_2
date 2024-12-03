import gradio as gr
import torch
import os
import csv
from datetime import datetime
from PIL import Image
from tempfile import NamedTemporaryFile
from transformers import (
    AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, 
    AutoModel, BitsAndBytesConfig, CLIPImageProcessor,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)
from gradio.themes.base import Base
import torchvision.transforms as T
import util
import prompt as pr
import requests
import argparse
import json

# ------------------------------------------------------------------------
# Run Setup
# ------------------------------------------------------------------------

def call_vqa_model(image_bytes, 
                   question, context, example, 
                   audience1, audience2, #audience3, 
                   audience4, audience_difficulty, 
                   formality, sentiment, language, 
                   model_choice, max_tokens, api_key, api_query):
    """
    Main function to call the VQA model based on user inputs.
    """
    print(question, context, example)

    # Grab context query from europeana
    #context_str = fetch_europeana_context(api_key, api_query)
    #print(context_str)

    image_bytes = get_image_bytes(image_bytes)
    prompt_dict = {
        'question': question,
        'context': context, 
        'example': example,
        'audience_age': audience1,
        'audience_abilities': audience2,
        #'audience_education': audience3,
        'audience_art': audience4,
        'formality': formality, 
        'sentiment': sentiment, 
        'language': language
    }
    with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_file.write(image_bytes)
        temp_file.seek(0)
        with Image.open(temp_file.name).convert('RGB') as raw_image:
            response = run_model(prompt_dict, raw_image, model_choice, max_tokens, img_path=temp_file, audience_difficulty=audience_difficulty)
            cleaned_response = util.clean_text(response)

    os.remove(temp_file.name)  # Clean up the temporary file
    prompt = pr.create_prompt(prompt_dict, audience_difficulty=audience_difficulty, raw=True)
    log_interaction(prompt_dict, cleaned_response, model_choice, max_tokens, audience_difficulty, prompt)
    return cleaned_response, gr.update(value=get_log_file())

def get_image_bytes(image_bytes):
    """
    Get image bytes from uploaded file (gradio upload field) or use default image (from path).
    """
    if not image_bytes:
        with open(DEFAULT_IMAGE_PATH, 'rb') as f:
            image_bytes = f.read()
    else:
        if isinstance(image_bytes, list) and image_bytes:
            image_bytes = image_bytes[0]  # Use the first image if it's a list
        elif not isinstance(image_bytes, bytes):
            raise ValueError("Expected a bytes-like object or a list of bytes-like objects for image_bytes")
    return image_bytes

def run_model(prompt_dict, raw_image, model_choice, max_tokens, img_path, audience_difficulty):
    """
    Run the selected model based on model_choice.
    """
    model_id = MODEL_CHOICES.get(model_choice)
    if model_choice == 'LLaVA-llama3-8b':
        return run_llava_llama(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty=audience_difficulty)    
    elif model_choice == 'LLaVA-gemma-7b':
        return run_llava_gemma(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty=audience_difficulty)
    elif model_choice == 'LLaVA-yi-34b':
        return run_llava_yi(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty=audience_difficulty)
    elif model_choice in ['InternVL', 'InternVL-Mini', 'InternVL2-8B']:
        return run_internvl(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty=audience_difficulty)
    elif model_choice == 'MiniCPM-Llama3-V-2_5':
        return run_minicpm(model_id, prompt_dict, img_path, max_tokens, audience_difficulty=audience_difficulty)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

def fetch_europeana_context(api_key, search_query, max_result):
    """
    Function to fetch Europeana context.
    """
    try:
        # Europeana API endpoint
        url = "https://api.europeana.eu/record/v2/search.json"
        params = {
            'wskey': api_key,
            'query': search_query,
            'start': 1,  # Start from the first result
            'rows': max_result,
        }

        # Fetch the Europeana API
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('totalResults', 0) == 0:
            return None, None, None, None, None, "No results found."
        item = data.get('items', [])[0]
        description = item.get('dcDescription', ['No description available'])[0]
        title = item.get('title', ['No Title'])[0]
        provider_url = item.get('guid', None)
        if not provider_url:
            return title, description, None, None, None, "Provider URL not found."

        # Fetch result metadata
        response = requests.get(provider_url, timeout=10)
        response.raise_for_status()
        metadata = fetch_metadata(response.content)
        image = fetch_image(response.content)

        return title, description, provider_url, metadata, image, None

    except requests.exceptions.RequestException as e:
        return None, None, None, None, f"Error fetching Europeana context: {e}"

def search_and_fetch_description(api_key, search_query, max_result=10):
    """Main function to tie everything together."""
    title, description, provider_url, metadata, img, error = fetch_europeana_context(api_key, search_query, max_result)
    if error:
        print(error)
        return
    print(f"Title: {title}")
    print(f"Description: {description}")
    print(f"Metadata: {metadata}")
    if provider_url:
        print(f"Fetching detailed description from: {provider_url}")
        detailed_description = fetch_object_description(provider_url)
        print(f"Detailed Description:\n{detailed_description}")
    else:
        print("No provider URL found.")
    return title + description + provider_url + metadata, img

def search_and_fetch_org_description(api_key, organization):
    """Main function to tie everything together."""
    dict_list = fetch_facets_with_values(api_key, organization)
    context = ''
    for item in dict_list:
        context += (
            f"ID: {item['id']}\n"
            f"Title: {item['title']}\n"
            f"Description: {item['description']}\n"
            f"Provider: {item['provider']}\n"
            f"URL: {item['sub_url']}\n"
            f"Metadata: {item['metadata']}\n\n"
        )
    print(f"Context: {context}")
    return context

# ------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------

def run_llava_llama(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty):
    """
    Run the LLaVA model.
    Source: https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-hf
    Licence: Apache-2.0 license
    """
        # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(model_id)
    prompt = pr.create_prompt(prompt_dict, audience_difficulty=audience_difficulty, raw=False)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    output = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        do_sample=False
    )
    return processor.decode(output[0][2:], skip_special_tokens=True).split("assistant", 1)[1].strip()

def run_llava_gemma(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty):
    """
    Run the LLaVA-gemma model.
    Source: https://huggingface.co/Intel/llava-gemma-7b
    Licence: https://ai.google.dev/gemma/terms

    """
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    processor = util.LlavaGemmaProcessor(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        image_processor=CLIPImageProcessor.from_pretrained(model_id)
    )

    prompt = processor.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f"<image>\n{pr.create_prompt(prompt_dict, audience_difficulty=audience_difficulty)}"}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16).to('cuda' if torch.cuda.is_available() else 'cpu')
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        #max_length=30
    )

    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

def run_llava_yi(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty):
    """
    Run the LLaVA model.
    Source: https://huggingface.co/llava-hf/llava-v1.6-34b-hf
    Licence: Apache-2.0 license
    """

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        #load_in_4bit=True # quantization
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    processor = LlavaNextProcessor.from_pretrained(model_id)

    # TODO: PROMPT
    prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"

    inputs = processor(prompt, raw_image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

    # autoregressively complete prompt
    output = model.generate(
        **inputs, 
        max_new_tokens=max_tokens
    )

    return processor.decode(output[0], skip_special_tokens=True)

def run_internvl(model_id, prompt_dict, raw_image, max_tokens, audience_difficulty):
    """
    Run the InternVL or InternVL-Mini model.
    Source: https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5
    Licence: MIT license
    """
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, quantization_config = quantization_config).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, skip_special_tokens=True)
    pixel_values = util.load_image(raw_image, max_num=6).to(torch.bfloat16).cuda()
    generation_config = dict(num_beams=1, max_new_tokens=max_tokens, do_sample=False)
    prompt = pr.create_prompt(prompt_dict, audience_difficulty=audience_difficulty, raw=True)
    responses = model.chat(
        tokenizer, 
        pixel_values, 
        # image_counts=pixel_values.size(0), 
        question=prompt, 
        generation_config=generation_config,
    )
    return responses

def run_minicpm(model_id, prompt_dict, img_path, max_tokens, audience_difficulty):
    """
    Run the MiniCPM-Llama3-V-2_5 model.
    Source: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
    Licence: Apache-2.0 license

    """
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        device_map='auto', 
        torch_dtype=torch.float16, 
        resume_download=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    
    prompt = pr.create_prompt(prompt_dict, audience_difficulty=audience_difficulty, raw=True)
    msgs = [{'role': 'user', 'content': prompt}]
    image = Image.open(img_path).convert('RGB')

    response = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default 
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )

    return response

# ------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------

def log_interaction(prompt_dict, output, model_choice, max_tokens, audience_difficulty, prompt):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'question': prompt_dict['question'],
        'context': prompt_dict['context'],
        'example': prompt_dict['example'],
        'audience_age': prompt_dict['audience_age'],
        'audience_abilities': prompt_dict['audience_abilities'],
        'audience_art': prompt_dict['audience_art'],
        'formality': prompt_dict['formality'],
        'sentiment': prompt_dict['sentiment'],
        'language': prompt_dict['language'],
        'model_choice': model_choice,
        'audience_difficulty': audience_difficulty,
        'max_tokens': max_tokens,
        'prompt': prompt,
        'output': output
    }
    
    # Open the file with utf-8 encoding
    with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_entry.keys())
        writer.writerow(log_entry)
    
def get_log_file():
    return log_file

# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------

def parse_args():
    """Returns: Command-line arguments"""
    parser = argparse.ArgumentParser('SMB/SPK VLM CLI')
    parser.add_argument('--imgpath', type=str, default="Kaufmann_Georg.png" , help='Path to the artwork', required=False)
    parser.add_argument('--cfgpath', type=str, help='Path to the .json containing required parameters', required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # log_file = 'logs/model_logs.csv'
    # with open(log_file, 'w', newline='') as csvfile:
    #     fieldnames = ['timestamp', 'question', 'context', 'audience', 'formality', 
    #                 'sentiment', 'language', 'model_choice', 'max_tokens', 'output']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()


    # audience_category_age = ["Childs", "Elderly", "Adults"]
    # audience_category_abilities = ["Visually Impaired", "ADHD", "Dyslexia"]
    # audience_category_art = ["Art-Professional", "Art-Hobbyist", "Art-Amateur"]
    # formality = ["Informal", "Formal"]
    # sentiment = ["Positive", "Neutral", "Negative"]
    # language = ["English", "Romanian", "German", "Serbian", "Hungarian", "Greek"]
    # MODEL_CHOICES = {
    #     'LLaVA-llama3-8b': "xtuner/llava-llama-3-8b-v1_1-transformers",
    #     #'LLaVA-gemma-7b': "Intel/llava-gemma-7b",
    #     #'LLaVA-yi-34b': "llava-hf/llava-v1.6-34b-hf",
    #     #'InternVL': "OpenGVLab/InternVL-Chat-V1-5",
    #     #'InternVL-Mini': "OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
    #     'InternVL2-8B': "OpenGVLab/InternVL2-8B",
    #     'MiniCPM-Llama3-V-2_5': 'openbmb/MiniCPM-Llama3-V-2_5'
    # }
    # europeana_api_key = "rrabsishan"


    # Specify the path to the image file
    with open(args.imgpath, "rb") as image_file:
        image_bytes = image_file.read()

    # Open and read the JSON file
    with open(args.cfgpath, 'r') as json_file:
        data = json.load(json_file)

    # Print the loaded data
    print(data)

    print(data['question'])


    # TODO

    text = call_vqa_model(image_bytes, data['question'], data['context'], data['example'], data['audience_age'], data['audience_abilities'], 
                          data['audience_art'], data['audience_difficulty'], data['formality'], data['sentiment'], 
                          data['language'], data['model_choice'], data['max_tokens'], data['api_key'], data['api_query'])







