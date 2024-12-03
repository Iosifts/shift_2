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
#from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Settings
#from llama_index.llms.huggingface import HuggingFaceLLM
#from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from gradio.themes.base import Base
from common_setup import create_common_layout, get_css
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import util
import prompt as pr
import requests
from bs4 import BeautifulSoup
from io import BytesIO

# ------------------------------------------------------------------------
# Run Setup
# ------------------------------------------------------------------------
# Create CSV log file
log_file = 'logs/model_logs.csv'
with open(log_file, 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'question', 'context', 'audience', 'formality', 
                  'sentiment', 'language', 'model_choice', 'max_tokens', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

DEFAULT_IMAGE_PATH = 'data/img/Kaufmann Georg.png'
MODEL_CHOICES = {
    'LLaVA-llama3-8b': "xtuner/llava-llama-3-8b-v1_1-transformers",
    #'LLaVA-gemma-7b': "Intel/llava-gemma-7b",
    #'LLaVA-yi-34b': "llava-hf/llava-v1.6-34b-hf",
    #'InternVL': "OpenGVLab/InternVL-Chat-V1-5",
    #'InternVL-Mini': "OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
    'InternVL2-8B': "OpenGVLab/InternVL2-8B",
    'MiniCPM-Llama3-V-2_5': 'openbmb/MiniCPM-Llama3-V-2_5'
}

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
    context_str = fetch_europeana_context(api_key, api_query)
    print(context_str)

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


#


def fetch_facets_with_values(api_key):
    """
    Fetch all available facets and combine them into a single list for a dropdown.
    """
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
        'wskey': api_key,
        'query': '*',  # Fetch all items
        'rows': 0,  # We don't need actual results, just facets
        'facet': 'DATA_PROVIDER'  # Adjust facets as needed
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("Response Data:", data)

        combined_options = []
        facets = data.get('facets', [])
        if not facets:
            print("No facets found in response.")
        exit(0)

        facets = data.get('facets', [])
        for facet in facets:
            facet_name = facet.get('name')
            for field in facet.get('fields', []):
                combined_options.append(f"{facet_name}: {field['label']}")
        print("combined_options", combined_options)
        return combined_options

    except requests.exceptions.RequestException as e:
        print(f"Error fetching facets from Europeana: {e}")
        return []

def fetch_items_for_selection(api_key, combined_selection):
    """
    Fetch items based on the combined selection of facet and value.
    """
    try:
        # Split the combined selection into facet name and value
        facet_name, selected_value = combined_selection.split(": ", 1)
        url = "https://api.europeana.eu/record/v2/search.json"
        params = {
            'wskey': api_key,
            'query': '*',
            'qf': f'{facet_name}:"{selected_value}"',
            'rows': 10,  # Adjust the number of items as needed
        }

        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('totalResults', 0) == 0:
            print("No results found.")
            return None

        i = 0
        dict_list = []
        for item in data.get('items', []):
            title = item.get('title', ['No Title'])[0]
            print("Title:", title)
            description = item.get('dcDescription', ['No description available'])[0]
            print("Description:", description)
            provider = item.get('edmDataProvider', 'Unknown')
            print("Provider:", provider)
            sub_url = item.get('guid', 'No URL')
            print("URL:", sub_url)
            print("----")
            
            if not sub_url:
                return "Provider URL not found."
            response = requests.get(sub_url, timeout=10)
            if response.status_code != 200:
                return f"Error: {response.status_code}"
            metadata = fetch_metadata(response.content)
            
            item_dict = {
                'id': i,
                'title': title,
                'description': description,
                'provider': provider,
                'sub_url': sub_url,
                'metadata': metadata
            }
            i += 1

            dict_list += item_dict
            
        return dict_list

    except requests.exceptions.RequestException as e:
        print(f"Error fetching items: {e}")
        return []

def fetch_org_results(api_key, search_query):

    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
        'wskey': api_key,
        'query': search_query,
        'qf': 'edmDataProvider:"Gemäldegalerie, Staatliche Museen zu Berlin"',  # Filter for the specified provider
        'start': 1,  # Start from the first result
        'rows': 1    # Fetch only one result
    }

    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        return None, None, None, f"Error: {response.status_code}"

    pass


#


def fetch_metadata(html_content):
    """
    Function to extract metadata from HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    h3_texts = [h3.get_text(strip=True) for h3 in soup.find_all('h3')]
    li_texts = []
    for ul in soup.find_all('ul'):
        li_items = [li.get_text(strip=True) for li in ul.find_all('li')]
        concatenated_li_text = ', '.join(li_items)
        if concatenated_li_text not in ['Home, Collections, Stories, Share your collections, Log in / Join', '']:
            li_texts.append(concatenated_li_text)
    metadata = '\n'
    for str1, str2 in zip(h3_texts, li_texts):
        metadata += str1 + ': ' + str2 + ',\n'
    return metadata

def fetch_image(html_content):
    """
    Function to extract the image download URL from HTML content using BeautifulSoup and download the image.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    download_link = soup.find('a', class_='download-button')
    if download_link and download_link.has_attr('href'):
        image_url = download_link['href']
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            return None
    else:
        print("Download link not found in the HTML content.")
        return None


#


def fetch_object_description(html_content):
    """
    Function to extract the image download URL from HTML content using BeautifulSoup and download the image.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    download_link = soup.find('a', class_='download-button')
    if download_link and download_link.has_attr('href'):
        image_url = download_link['href']
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            return None
    else:
        print("Download link not found in the HTML content.")
        return None

def fetch_object_description(object_url):
    """Function to fetch the object description from the SMB website."""
    try:
        # Fetch the webpage content
        response = requests.get(object_url, timeout=10)  # Timeout set to 10 seconds
        response.raise_for_status()  # Raise exception for bad HTTP status codes
    except requests.Timeout:
        return "Error: The request timed out."
    except requests.RequestException as e:
        return f"Error: An error occurred: {e}"
    pass


#


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

def log_interaction(prompt_dict, output, model_choice, max_tokens, audience_difficulty, prompt):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'question': prompt_dict['question'],
        'context': prompt_dict['context'],
        'example': prompt_dict['example'],
        'audience_age': prompt_dict['audience_age'],
        'audience_abilities': prompt_dict['audience_abilities'],
        #'audience_education': prompt_dict['audience_education'],
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

def update_dropdown(facet_name):
    """
    Update dropdown based on the selected facet.
    """
    if facet_name in facet_dict:
        return gr.update(choices=facet_dict[facet_name])
    return gr.update(choices=[])

def fetch_items(facet_name, selected_value):
    """
    Fetch items for the selected facet value.
    """
    return fetch_items_by_combined(europeana_api_key, facet_name, selected_value)

def fetch_combined_facets(api_key):
    """
    Fetch all available facets and combine them into a single list for a dropdown.
    """
    url = "https://api.europeana.eu/record/v2/search.json"
    params = {
        'wskey': api_key,
        'query': '*',  # Fetch all items
        'rows': 0,  # We don't need actual results, just facets
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        combined_options = []
        facets = data.get('facets', [])
        for facet in facets:
            facet_name = facet.get('name')
            for field in facet.get('fields', []):
                combined_options.append(f"{facet_name}: {field['label']}")

        return combined_options

    except requests.exceptions.RequestException as e:
        print(f"Error fetching facets from Europeana: {e}")
        return []

def fetch_items_by_combined(api_key, combined_selection):
    """
    Fetch items based on the combined selection of facet and value.
    """
    try:
        # Split the combined selection into facet name and value
        facet_name, selected_value = combined_selection.split(": ", 1)
        url = "https://api.europeana.eu/record/v2/search.json"
        params = {
            'wskey': api_key,
            'query': '*',
            'qf': f'{facet_name}:"{selected_value}"',
            'rows': 10,  # Adjust the number of items as needed
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        items = data.get('items', [])
        return [
            {
                "Title": item.get('title', ['No Title'])[0],
                "Description": item.get('dcDescription', ['No description available'])[0],
                "Provider": item.get('edmDataProvider', 'Unknown'),
                "URL": item.get('guid', 'No URL'),
            }
            for item in items
        ]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching items: {e}")
        return []

def single_select(selected, category):
    if len(selected) > 1:
        return selected[-1]  # Keep only the last selected option
    return selected

def multi_select(selected):
    return selected

def reset_inputs():
    # Directly return the default values for each input to clear them
    return ("",  # image_input
            "",  # question
            "",  # context
            "",  # example
            ["Adults"],  # audience1 (Age group)
            [],  # audience2 (Abilities)
            ["Normal Education"],  # audience3 (Education)
            ["Hobbyist"],  # audience4 (Art)
            "Medium",  # audience_difficulty
            "Formal",  # formality
            "Neutral",  # sentiment
            "English",  # language
            "LLaVA-llama3-8b",  # model_choice
            200,  # max_tokens
            "",  # Europeana API Key
            ""   # Europeana Search Query
           )

# ------------------------------------------------------------------------
# (Hardcoded) Gradio parameters
# ------------------------------------------------------------------------

audience_category_age = ["Childs", "Elderly", "Adults"]
audience_category_abilities = ["Visually Impaired", "ADHD", "Dyslexia"]
# audience_category_education = ["High Degree", "Standard Degree", "Literate"]
audience_category_art = ["Art-Professional", "Art-Hobbyist", "Art-Amateur"]

europeana_api_key = "rrabsishan"

max_result = 10

# TODO ValueError if No results found.

# TODO Filter by org results

# Fetch facets and items
# facet_value_options = fetch_facets_with_values(europeana_api_key)
# if facet_value_options:
#     combined_selection = facet_value_options[0]
#     items = fetch_items_for_selection(europeana_api_key, combined_selection)
#     facet_names = list(items.keys())
#     print("Fetched Items:", items)
# else:
#     print("No facets available.")

# TODO Display org result images in collection of some sort

# ------------------------------------
# Gradio Frontend
# ------------------------------------
css = get_css()
with gr.Blocks(css=css) as demo:
    create_common_layout()  # Apply common layout
    gr.Markdown("# Vision-Language Modelling", elem_id="title")
    gr.Markdown("## (1) Drop an Image to Describe", elem_id="description")
    with gr.Column():
        image_input = gr.Files(label="Upload Image (.jpg/.png)", type="binary", file_count='single')

    gr.Markdown("# Prompt String Creation ", elem_id="title")
    gr.Markdown("## (1) Text Input", elem_id="description")
    with gr.Row():
        # instruction prompting
        question = gr.Textbox(
            label="Additional Audience Question", 
            lines=10, 
            placeholder="E.g. 'What can you tell me about the image?' (optional)"
        )        
        # contextual prompting
        context = gr.Textbox(
            label="Image Context", 
            lines=10, 
            placeholder="E.g. Wikipedia text data of the painting or Information on the internet \
                about the painting (optional, recommended for optimal results)"
        )
        # few-shot prompting
        example = gr.Textbox(
            label="Example Description", 
            lines=10, 
            placeholder="E.g. 'The image XY, painted by Z, is a portrait of a woman.' Examples \
                can be multiple descriptions of different images. (optional, helpful for \
                the model to understand the task)"
        )


    with gr.Row():
        api_key_input = gr.Textbox(label="Europeana API Key", value="rrabsishan")
        search_query_input = gr.Textbox(label="Artwork title", placeholder="Enter the title of the painting.")
        fetch_context_button = gr.Button(value="Fetch Context from Europeana")

    #    




    image_output = gr.Image(label="Downloaded Image", type="pil", visible=True)
    fetch_context_button.click(
        search_and_fetch_description,
        inputs=[
            api_key_input, 
            search_query_input, 
            max_result
        ],
        outputs=[
            context, 
            image_output
        ]
    )

    gr.Markdown("## (2) Audience rules", elem_id="description")
    with gr.Row():

        with gr.Column():

            with gr.Row():
                audience_difficulty = gr.Dropdown(
                    label="Audience Prompt", allow_custom_value = False, 
                    info="The length of the prompt that contains audience rules constraints.",
                    choices=["Short", "Medium", "Long"], value="Medium")

                with gr.Column():
                    formality = gr.Dropdown(
                        label="Formality", choices=["Informal", "Formal"], 
                        value="Formal")
                    sentiment = gr.Dropdown(
                        label="Sentiment", choices=["Positive", "Neutral", "Negative"], 
                        value="Neutral")
                    language = gr.Dropdown(
                        label="Language", allow_custom_value = False, 
                        choices=["English", "Romanian", "German", "Serbian", "Hungarian", "Greek"], 
                        value="English")

                with gr.Column():
 
                    audience1 = gr.Dropdown(
                        interactive=True, 
                        label="Age", 
                        choices=audience_category_age, 
                        value="Adults",  # Default selected value
                        #elem_id="age-group"
                    )
                    audience1.change(single_select, inputs=audience1, outputs=audience1)

                    audience2 = gr.Dropdown(
                        interactive=True,
                        label="Restricted/Enhanced Abilities",
                        choices=audience_category_abilities,  # Pass the audience choices
                    )
                    audience2.change(multi_select, inputs=audience2, outputs=audience2)

                    # audience3 = gr.CheckboxGroup(
                    #     interactive=True,
                    #     label="Education", 
                    #     choices=audience_category_education,  # Pass the audience choices
                    #     value=["Normal Education"]  # Default selected value
                    # )
                    # audience3.change(single_select, inputs=audience3, outputs=audience3)

                    audience4 = gr.Dropdown(
                        interactive=True,
                        label="Art Education", 
                        choices=audience_category_art,  # Pass the audience choices
                        value=["Art-Hobbyist"]  # Default selected value
                    )
                    audience4.change(single_select, inputs=audience4, outputs=audience4)

    gr.Markdown("## (3) Model Selection and Manipulation", elem_id="description")
    with gr.Row():
        model_choice = gr.Dropdown(label="Choose Model", 
                                    choices=list(MODEL_CHOICES.keys()), 
                                    value="LLaVA-llama3-8b")
        max_tokens = gr.Slider(label="Max Tokens", minimum=50, maximum=1000, value=200, step=25)

    gr.Markdown("# Generate Description", elem_id="title")

    with gr.Row():
        submit_button = gr.Button(value="Detect Image", elem_classes="generate")
        reset_button = gr.Button("Reset", elem_id="reset_button")

    gr.Markdown("## Output", elem_id="description")
    with gr.Row():
        output_text = gr.Textbox(label="Output Text", lines=7, interactive=True)#, readonly=True)
        
    gr.Markdown("## Download Log File", elem_id="description")
    download_button = gr.File(label="Download Log File")

    submit_button.click(
        call_vqa_model, 
        inputs=[image_input,
                question, context, example,
                audience1, audience2, #audience3, 
                audience4, audience_difficulty,
                formality, sentiment, language, 
                model_choice, max_tokens,
                api_key_input, search_query_input], 
        outputs=[output_text, download_button]
    )
    reset_button.click (
        fn=reset_inputs, 
        inputs=[], 
        outputs=[image_input,
                question, context, example,
                audience1, audience2, #audience3, 
                audience4, audience_difficulty,
                formality, sentiment, language, 
                model_choice, max_tokens,
                api_key_input, search_query_input]  # Reset API inputs as well
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

demo.queue()
demo.launch(share=True)

