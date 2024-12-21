# TrOCR Trainer

This repo contains python scripts for training, inference, and data generation for Transformer-based Optical Character Recognition (OCR).

## Training:

You can download a dataset from here: https://drive.google.com/drive/folders/1Cm01jChTA63NOoM_9WkMBLkBy8XkxSAB?usp=drive_link \
Place the dataset in data/training and run the following
command to start the training. For hyperparameter tuning
check out the train.py script.

### Usage Example:
To run the Training, use the following command:
```sh
python train.py data/training/oscar_v1.12_5k
```

## Inference:
To run Inference on an image or .pdf using an existing huggingface or pytorch checkpoint, use the following command:

```sh
python inference.py Balcescu.png output/output_dir_with_checkpoint/best_checkpoint_char_acc.pt
```

The first parameter should be an image or .pdf. the second parameter points to a directory with a checkpoint.

## Generator:

The generator is a script relying on trdg package that generates images from a text file for Optical Character Recognition (OCR) training.

### Setup

Requires python 3.6 or lower, trdg package is not updated.

Backgrounds: 
- To use different backgrounds need to be added to each environments trdg/generators/images folder

Fonts:
- The usage of Fonts may cause an AttributeError 'FreeTypeFont' to appear. To fix this bug in the trdg package, simply replace the following:             
```sh
# modify trdg.utils.py:
    from PIL import ImageFont
    # replace get_text_height in utils.py:
    def get_text_height(font: ImageFont.FreeTypeFont, text: str) -> int:
        return font.getbbox(text)[3]
```
            
### Usage Example

To run the script, use the following command:

```sh
python generator.py path/to/input.txt path/to/output_dir --bytes_to_read 1000 --chunk_count 1 --chunk_length 100
```
Reads 10^{6} characters = 1 MB of text.\
Creates 10^{6} / (chunk_count * chunk_length) images.
Default: ~ 2 * 10^{3} images created. time: 10 minutes.

- input_path: Path to the input text file.
- output_dir: Directory to save the output images and labels.
- --bytes_to_read: Number of bytes to read from the input file. Default is 1 GiB (1,073,741,824 bytes).
- --chunk_count: Number of lines per image. It is refered to as chunks, since one page can have arbitrarily long lines of text. This can be fixated in the code about chunk lengths.
- --chunk_length: Maximum length of each text line/chunk. 

Output structure (running example):
The structure of data folder as below.
```
dataset
├── labels.csv
├── labels.txt
└── images
    ├── image0.png
    ├── image1.png
    ├── image2.png
    └── ...
```

### References:

### Image generation:
- https://github.com/Belval/TextRecognitionDataGenerator
### Training / finetuning:
Output format designed for EasyOCR, as explained here:
- https://github.com/clovaai/deep-text-recognition-benchmark

Alternative labeling schemes possible by using: --create_csv and adapting code for respective columns.
### General:
- https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/

