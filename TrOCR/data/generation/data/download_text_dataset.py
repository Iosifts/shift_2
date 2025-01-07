from datasets import load_dataset
from tqdm import tqdm
import requests
import os
import sys

def download_wiki(output_dir):
    """Download Romanian Wikipedia dump."""
    wiki_url = "https://dumps.wikimedia.org/rowiki/latest/rowiki-latest-pages-articles-multistream.xml.bz2"
    output_file = os.path.join(output_dir, "rowiki-articles.xml.bz2")
    
    if os.path.exists(output_file):
        print(f"Wiki dump already exists at {output_file}")
        return output_file
        
    print("Downloading Wikipedia dump...")
    response = requests.get(wiki_url, stream=True)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to download Wiki dump. Status code: {response.status_code}")
        
    total_size = int(response.headers.get('content-length', 0))
    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    return output_file

def download_oscar(output_dir, preview_chars=2000):
    """Download Romanian OSCAR dataset."""
    output_file = os.path.join(output_dir, "oscar_text.txt")
    
    if os.path.exists(output_file):
        print(f"OSCAR dataset already exists at {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            preview = f.read(preview_chars)
            print(f"\nPreview ({preview_chars} chars):\n{preview}")
        return output_file
    
    print("Downloading OSCAR dataset...")
    try:
        dataset = load_dataset("oscar", "unshuffled_deduplicated_ro")
        train_split = dataset['train']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(train_split, desc="Writing OSCAR dataset"):
                f.write(example['text'] + "\n")
                
    except Exception as e:
        print(f"Error downloading OSCAR dataset: {e}")
        raise
    
    return output_file

def main():
    # Create output directory
    output_dir = os.path.join("data", "generation", "data", "input")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        wiki_file = download_wiki(output_dir)
        print(f"Wikipedia dump saved to: {wiki_file}")
        
        oscar_file = download_oscar(output_dir)
        print(f"OSCAR dataset saved to: {oscar_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
