import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
import time
import csv
import string
import argparse
from tqdm import tqdm
import numpy as np
import albumentations as A
from fontTools.ttLib import TTFont
import glob
import logging
import shutil
from data.generation.languages import get_available_languages, load_language_patterns
logging.getLogger("fontTools.ttLib.tables._p_o_s_t").setLevel(logging.ERROR)

def generate(args):
    """
    Generates text images using PIL ImageDraw and ImageFont.
    """
    start_time = time.time()

    # Open file and keep it open for the entire generation process
    file = open(args.input_path, 'r', encoding='utf-8')
    
    try:
        # Get file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to start
        
        # Estimate bytes needed based on target images
        # Average chars per chunk * chunks per image * number of images
        estimated_chars_needed = args.max_length * args.sentences_per_page * args.image_count
        estimated_bytes_needed = estimated_chars_needed * 2  # Rough estimate, 2 bytes per char
        
        if estimated_bytes_needed > file_size:
            raise ValueError(
                f"Input file too small for {args.image_count} images. "
                f"Need ~{estimated_bytes_needed:,} bytes, but file is {file_size:,} bytes. "
                f"Either reduce args.image_count or use a larger input file."
            )
        
        text = file.read(estimated_bytes_needed)

        # Load language patterns
        language_patterns = load_language_patterns(args.language)
        if not language_patterns:
            print(f"No patterns found for language '{args.language}'. Using basic text splitting.")
        
        text_chunks = split_text(text, wordbased=args.wordbased,
                            min_length=args.min_length, 
                            max_length=args.max_length,
                            language_patterns=language_patterns)
        
        # Check if we have enough chunks
        required_chunks = args.image_count * args.sentences_per_page
        if len(text_chunks) < required_chunks:
            raise ValueError(
                f"Not enough text chunks for {args.image_count} images. "
                f"Need {required_chunks:,} chunks but only got {len(text_chunks):,}. "
                f"Either reduce args.image_count or increase bytes_to_read."
            )

        # Limit chunks to what we need
        text_chunks = text_chunks[:required_chunks]

        # Creates new output directory and deletes existing one
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

        # Output directories
        image_dir_path = os.path.join(args.output_dir, 'images')
        os.makedirs(image_dir_path, exist_ok=True)
        labels_txt_path = os.path.join(args.output_dir, 'labels.txt')

        # Initialize CSV if needed
        if args.create_csv:
            csv_file = open(os.path.join(args.output_dir, 'labels.csv'), 'w', newline='', encoding='utf-8')
            writer = csv.writer(csv_file)
            writer.writerow(['img_path', 'text'])
        else:
            csv_file = None
            writer = None

        # Get supported fonts
        supported_fonts = get_random_font(args.font_dir, args.characters)
        if not supported_fonts:
            print("No fonts support the required Romanian characters.")
            return

        # Create a progress bar for the overall generation
        pbar = tqdm(total=args.image_count, desc="Generating images")
        k = 0  # Counter for saved images
        attempts = 0  # Counter for generation attempts
        
        with open(labels_txt_path, 'w', encoding='utf-8') as f:
            while k < args.image_count:
                attempts += 1
                
                # Check if we need more text chunks
                if len(text_chunks) < args.sentences_per_page:
                    more_text = file.read(estimated_bytes_needed)
                    if more_text:
                        more_chunks = split_text(more_text, wordbased=args.wordbased,
                                              min_length=args.min_length, max_length=args.max_length)
                        text_chunks.extend(more_chunks)
                    else:
                        # If no more text, rewind file and start over
                        file.seek(0)
                        more_text = file.read(estimated_bytes_needed)
                        more_chunks = split_text(more_text, wordbased=args.wordbased,
                                              min_length=args.min_length, max_length=args.max_length)
                        text_chunks.extend(more_chunks)
                
                # Create a temporary list for the current page's images
                temp_images = []
                temp_texts = []
                
                for i in range(args.sentences_per_page):
                    if not text_chunks:  # If we've used all chunks, break
                        break
                        
                    chunk = text_chunks.pop(0)
                    
                    # Modify image width calculation to be more proportional to text length
                    base_width = 512  # minimum width
                    chars_per_pixel = 10  # approximate number of characters that fit in base_width
                    estimated_width = max(base_width, int(len(chunk) * (base_width / chars_per_pixel)))
                    image_width = min(1024, estimated_width)  # cap at 1024px
                    image_height = 64
                    
                    # Create image
                    img = Image.new('RGB', (image_width, image_height), color='white')
                    if args.background_dir:
                        # Get and prepare background first
                        bg_img = get_random_background(args.background_dir)
                        if bg_img:
                            # Resize to cover the entire area while maintaining aspect ratio
                            bg_aspect = bg_img.width / bg_img.height
                            target_aspect = image_width / image_height
                            
                            # Calculate dimensions that will cover the entire target area
                            if bg_aspect > target_aspect:
                                # Background is wider, scale to match height
                                new_height = image_height
                                new_width = int(new_height * bg_aspect)
                            else:
                                # Background is taller, scale to match width
                                new_width = image_width
                                new_height = int(new_width / bg_aspect)
                            
                            bg_img = bg_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            # Center crop to target size
                            left = (new_width - image_width) // 2
                            top = (new_height - image_height) // 2
                            bg_img = bg_img.crop((
                                left,
                                top,
                                left + image_width,
                                top + image_height
                            ))
                            
                            # Use the cropped background
                            img = bg_img.copy()

                    # Create a transparent layer for text
                    text_layer = Image.new('RGBA', (image_width, image_height), color=(255, 255, 255, 0))
                    draw = ImageDraw.Draw(text_layer)

                    # Calculate font size and position
                    font_path = random.choice(supported_fonts)
                    font_size = calculate_font_size(draw, chunk, font_path, image_width, image_height)
                    font = ImageFont.truetype(font_path, font_size)

                    # Center text with more precise calculations
                    text_bbox = draw.textbbox((0,0), chunk, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Calculate center position including the font's baseline offset
                    x = (image_width - text_width) // 2
                    y = (image_height - text_height) // 2 - text_bbox[1]  # Adjust for baseline

                    # Draw text on transparent layer
                    draw.text((x, y), chunk, font=font, fill=(0, 0, 0, 230))

                    # Composite text layer onto background
                    img.paste(text_layer, (0, 0), text_layer)

                    # Apply enhancements and augmentations
                    img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 1.8))
                    img = ImageEnhance.Contrast(img).enhance(random.uniform(1.2, 1.8))

                    # Apply albumentations
                    img_array = np.array(img)
                    augmented = define_augmentations()(image=img_array)
                    augmented_img = Image.fromarray(augmented['image'].astype(np.uint8))
                        
                    # Only add to temp lists if image generation was successful
                    if not has_black_bars(augmented_img):
                        temp_images.append(augmented_img)
                        temp_texts.append(chunk)  # Store the text
                
                # Only save if we have valid images
                if temp_images:
                    # Combine the valid images
                    max_width = max(img.width for img in temp_images)
                    line_height = max(img.height for img in temp_images)
                    total_height = len(temp_images) * line_height
                    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                    current_height = 0

                    for img in temp_images:
                        combined_image.paste(img, (0, current_height))
                        current_height += line_height

                    # Final check on combined image
                    if not has_black_bars(combined_image):
                        # Save image first
                        img_filename = f'image{k}.png'
                        full_img_path = os.path.join(image_dir_path, img_filename)
                        combined_image.save(full_img_path)

                        # Then write the labels for this image
                        combined_text = ' '.join(temp_texts)  # Combine all texts for this image
                        f.write(f'{img_filename}\t{combined_text}\n')
                        if args.create_csv:
                            writer.writerow([img_filename, combined_text])

                        k += 1
                        pbar.update(1)

    finally:
        # Make sure we close the file even if there's an error
        file.close()

    pbar.close()  # Close progress bar
    
    # Close CSV file if it exists
    if csv_file:
        csv_file.close()

    print(
        'Generation time:', int(time.time() - start_time), 'seconds',
        '\nImage count:', k,
        '\nAttempts:', attempts
    )

def split_text(text, wordbased=False, min_length=1, max_length=100, language_patterns=None):
    """
    Splits text into chunks with awareness of language patterns and natural text flow.
    
    Args:
        text (str): Input text to split
        wordbased (bool): If True, split into individual words
        min_length (int): Minimum length of each chunk
        max_length (int): Maximum length of each chunk
        language_patterns (dict): Dictionary of language-specific patterns to preserve
    
    Returns:
        list: List of text chunks
    """
    # Handle word-based mode
    if wordbased:
        return [w for w in text.split() if min_length <= len(w) <= max_length]

    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        # Try to match language patterns first
        if language_patterns:
            pattern_found = False
            for pattern_list in language_patterns.values():
                for pattern in pattern_list:
                    pattern_words = pattern.split()
                    if i + len(pattern_words) <= len(words):
                        potential_match = ' '.join(words[i:i+len(pattern_words)])
                        if potential_match.startswith(pattern):
                            if min_length <= len(potential_match) <= max_length:
                                chunks.append(potential_match)
                            i += len(pattern_words)
                            pattern_found = True
                            break
                if pattern_found:
                    break
            if pattern_found:
                continue

        # Determine chunk size
        if random.random() < 0.7:  # 70% chance for 1-2 words
            chunk_size = random.randint(1, 2)
        else:  # 30% chance for 3 words
            chunk_size = 3

        # Create and validate chunk
        end_idx = min(i + chunk_size, len(words))
        chunk = ' '.join(words[i:end_idx])
        
        # Handle oversized chunks
        while len(chunk) > max_length and end_idx > i + 1:
            end_idx -= 1
            chunk = ' '.join(words[i:end_idx])
        
        # If single word is too long, skip it
        if len(chunk) > max_length:
            i += 1
            continue
            
        # Add valid chunk if it meets length requirements
        if min_length <= len(chunk) <= max_length:
            chunks.append(chunk)
        
        i = end_idx

    return chunks

def can_render_chars(font_path, characters):
    """
    Check if the font can render the specified characters.
    
    Args:
        font_path (str): The path to the font file.
        characters (str): The characters to check for support.
        
    Returns:
        bool: True if the font can render all specified characters, False otherwise.
        list: A list of characters that the font is missing.
    """
    font = TTFont(font_path)
    if 'cmap' not in font:
        return False, characters
    cmap = font['cmap'].getBestCmap()
    if cmap is None:
        # This font's cmap is not compatible or empty; skip it.
        return False, characters

    missing_chars = [char for char in characters if ord(char) not in cmap]
    return (len(missing_chars) == 0), missing_chars

def get_random_background(background_dir):
    """
    Get a random background image from the specified directory.
    
    Args:
        background_dir (str): Directory containing background images
    Returns:
        PIL.Image: Background image
    """
    backgrounds = glob.glob(os.path.join(background_dir, '*.[jp][pn][g]'))  # Get .jpg, .png, .jpeg
    if not backgrounds:
        return None
    bg_path = random.choice(backgrounds)
    return Image.open(bg_path).convert('RGB')

def get_random_font(font_dir, test_characters):
    """
    Get a random font that supports the specified characters.
    
    Args:
        font_dir (str): The directory containing font files.
        test_characters (str): The characters to test for support.
        
    Returns:
        list: A list of paths to font files that support the specified characters.
    """
    supported_fonts = []
    fonts = [os.path.join(font_dir, font) for font in os.listdir(font_dir) if font.endswith('.ttf')]
    for font_path in fonts:
        try:
            can_render, missing_chars = can_render_chars(font_path, test_characters)
            if can_render:
                supported_fonts.append(font_path)
            else:
                 #print(f"Font {font_path} is missing characters: {''.join(missing_chars)}")
                 pass
        except Exception as e:
            #print(f"Error with font {font_path}: {e}")
            pass

    print(f'{len(supported_fonts)}/{len(fonts)} supported fonts.')
    
    return supported_fonts

def calculate_font_size(draw, sentence, font_path, image_width, image_height, max_font_size=64):
    """
    Calculate optimal font size with better proportions
    """
    font_size = max_font_size
    min_font_size = 24  # Set minimum font size to prevent too small text

    while font_size >= min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0,0), sentence, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Check if text fits within 90% of the image dimensions
        if text_width <= image_width * 0.9 and text_height <= image_height * 0.8:
            break

        font_size -= 1

    return max(font_size, min_font_size)

def define_augmentations():
    return A.Compose([
        # Subtle color shifts to simulate aged paper
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7),
            A.ToSepia(p=0.3),  # Gives an aged look
        ], p=0.8),

        # Very subtle distortions to simulate paper texture
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),  # Subtle noise
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),  # Subtle warping
        ], p=0.5),
    ])

def has_black_bars(img, threshold=0.95):
    """
    Check if image has black bars by looking at the top and bottom rows.
    Returns True if black bars are detected.
    
    Args:
        img: PIL Image
        threshold: darkness threshold (0-1), higher means darker
    """
    # Convert to numpy array for easier analysis
    img_array = np.array(img)
    
    # Check top and bottom 3 rows
    top_rows = img_array[:3, :, :]
    bottom_rows = img_array[-3:, :, :]
    
    # Calculate average darkness
    def is_dark(pixels):
        return np.mean(pixels) < (255 * (1 - threshold))
    
    # Check if either top or bottom has dark bars
    has_top_bar = is_dark(top_rows)
    has_bottom_bar = is_dark(bottom_rows)
    
    return has_top_bar or has_bottom_bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text images")
    parser.add_argument("input_path", type=str, help="Path to the input text file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output images and labels.")
    parser.add_argument("--image_count", type=int, default=50, help="Number of resulting images.")
    parser.add_argument("--sentences_per_page", type=int, default=50, help="Number of (stacked) text-rows per page.")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sentence length.")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum sentence length.")
    parser.add_argument("--background_dir", type=str, default='data/generation/assets/backgrounds', help="Directory for background images.")
    parser.add_argument("--font_dir", type=str, default='data/generation/assets/fonts/combined', help="Directory for font files.")
    parser.add_argument("--characters", type=str, default=string.printable[:-6] + "ăâîșțĂÂÎȘȚ" + " ", help="Characters to support.")
    parser.add_argument("--create_csv", type=bool, default=True, help="Whether to create a CSV file with labels.")
    parser.add_argument("--wordbased", type=bool, default=False, help="One word per image mode")

    available_languages = get_available_languages()
    parser.add_argument(
        "--language",
        type=str,
        choices=available_languages,
        default='ro',
        help="Language code for text patterns. Available: " + ", ".join(available_languages)
    )

    args = parser.parse_args()

    generate(args)
