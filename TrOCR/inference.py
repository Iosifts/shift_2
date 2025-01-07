# -*- coding: utf-8 -*-
import os
import re
import sys
import string
import torch
import torch.utils.data
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import logging
logging.set_verbosity_error()  # Only errors will be printed
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
import pdfplumber
import easyocr
import warnings
import util
import argparse
import json

def extract_text_ckpt(model, processor, image_bytes, draw, mode='img'):
    """
    Extract text and bounding boxes from image or PDF using an OCR model and bounding-box generation.
    """
    if mode == 'img':
        image_parts = create_images_from_bounding_boxes(image_bytes, draw)
    elif mode == 'pdf':
        image_parts = create_images_from_pdf_bounding_boxes_plumber(image_bytes, draw)
    else:
        raise ValueError("Invalid mode. Use 'img' or 'pdf'.")

    full_text = []

    for part in image_parts:
        # Convert each cropped part to pixel values, then run OCR model
        pixel_values = processor(part, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        clean_text = generated_text.strip()
        full_text.append(clean_text + "\n")

    return clean_whitespaces("".join(full_text)), image_parts

def create_images_from_bounding_boxes(image_bytes, draw, buffer=5):
    """
    Extracts word/sentence bounding-boxes from an image using easyocr reader
    """
    image = Image.open(BytesIO(image_bytes))
    results = reader.readtext(image_bytes, detail=1)
    bounding_boxes = [result[0] for result in results]

    if draw:
        image_with_boxes = draw_bounding_boxes(image_bytes, bounding_boxes)
        image_with_boxes.show()

    parts = []

    for box in bounding_boxes:
        left = min(point[0] for point in box) - buffer
        top = min(point[1] for point in box) - buffer
        right = max(point[0] for point in box) + buffer
        bottom = max(point[1] for point in box) + buffer
        left = max(0, left)
        top = max(0, top)
        right = min(image.width, right)
        bottom = min(image.height, bottom)
        part = image.crop((left, top, right, bottom))
        parts.append(part)
    
    return parts

def create_images_from_pdf_bounding_boxes_plumber(pdf_stream, draw, buffer=5):
    """
    Extracts bounding boxes for words or sentences from a PDF file using pdfplumber
    and returns cropped images for each box.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to process (0-indexed).
        draw (bool): If True, draw bounding boxes on a copy of the page image.
        buffer (int): Buffer in pixels around each bounding box.

    Returns:
        List of cropped images for each bounding box.
    """
    parts = []
    pdf_stream = BytesIO(pdf_stream)
    
    with pdfplumber.open(pdf_stream) as pdf:
        page = pdf.pages[0]
        page_image = page.to_image()
        words = page.extract_words()

        if draw:
            for word in words:
                x0 = max(0, word['x0'] - buffer)
                y0 = max(0, word['top'] - buffer)
                x1 = min(page.width, word['x1'] + buffer)
                y1 = min(page.height, word['bottom'] + buffer)
                rect_width = x1 - x0
                rect_height = y1 - y0
                if rect_width > 0 and rect_height > 0:
                    # Pass top-left and bottom-right coordinates to draw_rect
                    page_image.draw_rect((x0, y0, x1, y1), fill=None, stroke="red", stroke_width=1)
                else:
                    print(f"Skipped drawing invalid box with coordinates: x0={x0}, y0={y0}, x1={x1}, y1={y1}")

            page_image.show()

        image = page_image.original
        for word in words:
            x0 = max(0, word['x0'] - buffer)
            y0 = max(0, word['top'] - buffer)
            x1 = min(page.width, word['x1'] + buffer)
            y1 = min(page.height, word['bottom'] + buffer)
            if (x1 > x0) and (y1 > y0):
                part = image.crop((x0, y0, x1, y1))
                parts.append(part)
            else:
                print(f"Skipped cropping invalid box with coordinates: x0={x0}, y0={y0}, x1={x1}, y1={y1}")

    return parts

def draw_bounding_boxes(image_bytes, bounding_boxes):
    """
    Draws the bounding boxes on an image
    """
    image = Image.open(BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    
    for box in bounding_boxes:
        left = min([point[0] for point in box])
        upper = min([point[1] for point in box])
        right = max([point[0] for point in box])
        lower = max([point[1] for point in box])
        draw.rectangle([left, upper, right, lower], outline="red", width=2)
    
    return image

def clean_whitespaces(text):
    """
    Remove leading and trailing whitespaces
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def visualize_image_parts_lines(image_parts):
    """
    Visualize cropped image parts for debugging/verification.
    """
    if not image_parts:
        print("No image parts to visualize.")
        return

    plt.figure(figsize=(10, 20))
    for i, part in enumerate(image_parts):
        plt.subplot(len(image_parts), 1, i+1)
        plt.imshow(part)
        plt.axis('off')
    plt.show()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='OCR Inference')
    
    # Required arguments
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image or PDF file')
    
    # Optional arguments with defaults
    parser.add_argument('--model_path', type=str, 
                       default="microsoft/trocr-large-handwritten",
                       help='Path to HuggingFace model or checkpoint')
    
    parser.add_argument('--checkpoint_path', type=str, 
                       default=None,
                       help='Path to .pt checkpoint file')
    
    parser.add_argument('--reference_text_path', type=str,
                       default='data/reference_texts/sample.txt',
                       help='Path to reference text file for comparison')
    
    parser.add_argument('--predictions_file', type=str,
                       default='data/output/oscar_v2_50k/trocr-large-handwritten/e20_lr1e-06_b4_fr1.0_tfr1.0_balcesu_test/predictions/test_preds_e2.json',
                       help='Path to predictions JSON file')
    
    parser.add_argument('--draw', action='store_true', default=True,
                       help='Enable drawing bounding boxes on input for visualization')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reader = easyocr.Reader(['ro'])

    # Validate input file
    if not os.path.isfile(args.image_path):
        print(f"Image/PDF file {args.image_path} does not exist.")
        sys.exit(1)
    else:
        with open(args.image_path, 'rb') as file:
            file_bytes = file.read()

    # Checkpoint loading
    if args.model_path is None and args.checkpoint_path is None:
        raise ValueError("No Hugging Face model or checkpoint provided.")
        
    if os.path.isdir(args.model_path):
        # Load processor and model from Hugging Face directory
        processor = TrOCRProcessor.from_pretrained(args.model_path)
        model = VisionEncoderDecoderModel.from_pretrained(args.model_path).to(device)
    elif args.checkpoint_path and os.path.isfile(args.checkpoint_path) and args.checkpoint_path.endswith('.pt'):
        # Load a default Hugging Face model and update it with the checkpoint
        print(f"Loading .pt model {args.model_path} and updating with checkpoint...")
        processor = TrOCRProcessor.from_pretrained(args.model_path)  # Default model
        model = VisionEncoderDecoderModel.from_pretrained(args.model_path).to(device)
        
        # Add special tokens used during training
        special_chars = ["ă", "â", "î", "ș", "ț", "Ă", "Â", "Î", "Ș", "Ț"]
        processor.tokenizer.add_tokens(special_chars, special_tokens=False)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
        # Load checkpoint
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            
        # Extract state_dict from checkpoint
        checkpoint_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Adjust mismatched embedding sizes
        state_dict = model.state_dict()
        for key in ["decoder.model.decoder.embed_tokens.weight", "decoder.output_projection.weight"]:
            if key in checkpoint_state_dict:
                if checkpoint_state_dict[key].size() != state_dict[key].size():
                    print(f"Resizing {key} from {checkpoint_state_dict[key].size()} to {state_dict[key].size()}")
                    checkpoint_state_dict[key] = torch.nn.functional.pad(
                        checkpoint_state_dict[key],
                        (0, 0, 0, state_dict[key].size(0) - checkpoint_state_dict[key].size(0))
                    )
        # Load the updated state_dict
        model.load_state_dict(checkpoint_state_dict)
    else:
        print("Invalid path provided. It must be either a Hugging Face model directory or a .pt checkpoint file.")
        sys.exit(1)

    # Inference
    if args.image_path.lower().endswith('.pdf'):
        extracted_text, image_parts = extract_text_ckpt(model, processor, 
                                                      file_bytes, args.draw, mode='pdf')
    elif args.image_path.lower().endswith(('.jpg', '.png')):
        extracted_text, image_parts = extract_text_ckpt(model, processor, 
                                                      file_bytes, args.draw, mode='img')
    else:
        print("Unsupported file format. Please provide a .pdf, .jpg, or .png file.")
        sys.exit(1)

    print("Recognized Text:", extracted_text)

    # Load reference text if provided
    if os.path.exists(args.reference_text_path):
        with open(args.reference_text_path, 'r', encoding='utf-8') as f:
            reference_text = f.read()
        metrics = util.compute_all_metrics(extracted_text, reference_text)
        print("\nMetrics against reference text:")
        print(metrics)
        print("\nGroundtruth:", reference_text)

    # Load and process predictions if provided
    if os.path.exists(args.predictions_file):
        with open(args.predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
        
        metrics = util.compute_json_metrics(args.predictions_file)
        
        all_predictions = clean_whitespaces(" ".join([pred['prediction'] for pred in predictions_data['predictions']]))
        all_references = clean_whitespaces(" ".join([pred['reference'] for pred in predictions_data['predictions']]))
        
        print("\nConcatenated Predictions:")
        print(all_predictions)
        print("\nConcatenated References:")
        print(all_references)
        
        print("\nMean Metrics:")
        for metric_name, value in metrics['mean_metrics'].items():
            print(f"{metric_name}: {value:.4f}")
        
        concat_metrics = util.compute_metrics(all_predictions, all_references)
        print("\nMetrics on Concatenated Text:")
        for metric_name, value in concat_metrics.items():
            print(f"{metric_name}: {value:.4f}")
