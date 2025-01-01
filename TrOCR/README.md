# Transformer-Based OCR

This repository offers a training pipeline for transformer-based text detection (OCR) models.

---

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Data

Currently two dataset formats are accepted - IAM words and any dataset 
correctly fitting the custom dataset class.

1. Download the public benchmark dataset IAM words:
https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database

2. Download an old document example dataset from google drive in romanian language:
https://drive.google.com/drive/folders/1Cm01jChTA63NOoM_9WkMBLkBy8XkxSAB?usp=drive_link \
This data was acquired and labeled using a synthetic neural network for handwritten
text generation (see generator.py)

Please refer to dataset.py for file structure

## 🧪 Training

Place the dataset in data/datasets and run the following
command to start the training. For hyperparameter tuning
check out the train.py script.

### Usage Example:

To run the Training, use the following command:
```sh
python train.py --data data/datasets/oscar_v1.12_5k --epochs 5
```

To use different dataset
```sh
python train.py --data data/datasets/iam_words --epochs 5 --dataset IAM
```

To use different model
```sh
python train.py --data data/datasets/oscar_v1.12_5k --epochs 5 --model naver-clova-ix/donut-base --batchsize 1

python train.py --data data/datasets/oscar_v1.12_5k --epochs 5 --model facebook/nougat-base --batchsize 1
```

To continue training from checkpoint:
```sh
python train.py --data data/datasets/oscar_v1.12_5k --checkpoint data\output\oscar_v1.12_5k\trocr-large-handwritten\e20_lr1e-06_b4_1222\best_checkpoint.pt --epochs 20
```

And provide a config.yaml for easier handling:
```sh
python train.py --config data/configs/config.yaml
```

## Arguments for `train.py`

Below is a detailed explanation of all arguments available in the `train.py` script:

### Positional Arguments
- **`data`** *(str)*:  
  Path to the directory containing labeled image data for training. This is a required argument.

### Optional Arguments
- **`--evaldata`** *(str)*:  
  Path to the directory containing labeled image data for evaluation.  
  Default: `data/datasets/balcesu_test`.

- **`--output`** *(str)*:  
  Directory where checkpoints, logs, and other outputs will be saved.  
  Default: `data/output`.

- **`--checkpoint`** *(str)*:  
  Path to an existing checkpoint `.pt` file to resume training from. If not provided, training starts from scratch.  
  Default: `None`.

- **`--model`** *(str)*:  
  Specifies the pre-trained model to use. Available options are:
  - `microsoft/trocr-base-stage1`
  - `microsoft/trocr-large-stage1`
  - `microsoft/trocr-base-handwritten`
  - `microsoft/trocr-large-handwritten`
  - `custom` (For custom models)  
  Default: `microsoft/trocr-large-handwritten`.

- **`--dataset`** *(str)*:  
  Specifies the dataset type. Available options are:
  - `IAM`
  - `custom` (For custom datasets, refer to `dataset.OCRDataset`)  
  Default: `custom`.

- **`--epochs`** *(int)*:  
  Number of epochs to train the model.  
  Default: `5`.

- **`--batchsize`** *(int)*:  
  Batch size for the DataLoader.  
  Default: `4`.

- **`--val_iters`** *(int)*:  
  Specifies how frequently (in epochs) the evaluation is performed during training.  
  Default: `1`.

- **`--lr`** *(float)*:  
  Learning rate for the optimizer during the update step.  
  Default: `1e-6`.

- **`--lr_patience`** *(int)*:  
  Number of epochs to wait before reducing the learning rate if no improvement is observed.  
  Default: `2`.

- **`--num_samples`** *(float)*:  
  Number of sample predictions to print during evaluation.  
  Default: `10`.

### Example Commands

- **Start training a model:**
  ```sh
  python train.py data/datasets/oscar_v1.12_5k --epochs 5

## Inference:
To run Inference on an image or .pdf using an existing huggingface or pytorch checkpoint, use the following command:

```sh
python inference.py Balcescu.png output/output_dir_with_checkpoint/best_checkpoint_char_acc.pt
```

The first parameter should be an image or .pdf. the second parameter points to a directory with a checkpoint.

## 📦 Generator:

The generator is a script relying on trdg package that generates images from a text file for Optical Character Recognition (OCR) training.
The output of the generator script fits the data structure of the train.py script.

### Setup

Requires python 3.6 or lower, trdg package is not updated. A separate execution environment is therefore recommended to run generator.py.

Backgrounds: 
- To use different backgrounds need to be added to each environments trdg/generators/images folder

Fonts: More than thousands of fonts can be used to inject more variance in the distribution of the dataset
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
- input_path: Path to the input text file.
- output_dir: Directory to save the output images and labels.
- --bytes_to_read: Number of bytes to read from the input file. Default is 1 GiB (1,073,741,824 bytes).
- --chunk_count: Number of lines one the final output page. If 1, then just one string of text.
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

## 👤 Acknowledgments:

### Image generation:
- https://github.com/Belval/TextRecognitionDataGenerator
### Training / finetuning:
Output format designed for EasyOCR, as explained here:
- https://github.com/clovaai/deep-text-recognition-benchmark

Alternative labeling schemes possible by using: --create_csv and adapting code for respective columns.
### General:
- https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/

