import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import logging
logging.set_verbosity_error()  # Only errors will be printed

from PIL import Image
import sys
import os

def infer_ocr(image_path, model_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the processor and model
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    # Read the image
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Image file {image_path} does not exist.")
        sys.exit(1)

    # provide outputdir path:
    model_dir = 'output/output_20241209-170159'
    text = infer_ocr(image_path, model_dir)
    print("Recognized text:", text)
