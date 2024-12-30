# Transformer-Based OCR

<div align="center">
  <img src="data/logo.png" alt="Transformer-Based OCR Logo" width="800"/>
</div>

Welcome to the **Transformer-Based OCR** repository! This project leverages state-of-the-art transformer architectures to accurately recognize and extract text from images and documents. With a focus on adaptability, scalability, and precision, this model is ideal for applications like document digitization, handwriting recognition, and more.

---

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Data Structure

You can download a romanian language dataset from here to test: https://drive.google.com/drive/folders/1Cm01jChTA63NOoM_9WkMBLkBy8XkxSAB?usp=drive_link \
This repository assumes the following structure of dataset:
```bash
> tree dataset_name
dataset_name
├── images
│   ├── image0.png
│   ├── image1.png
│   ├── image2.png
│             .
│             .
├── labels.txt

> cat labels.txt
image0.png	12
image1.png	Abia
image2.png	astăzi
image3.png	însă,
image4.png	două-zeci
image5.png	și
image6.png	cinci
image7.png	de
image8.png	ani
image9.png	după
     .
     .
```

## 🧪 Training

Place the dataset in data/training and run the following
command to start the training. For hyperparameter tuning
check out the train.py script.

### Usage Example:
To run the Training, use the following command:
```sh
python train.py data/training/oscar_v1.12_5k --epochs 5
```

To continue training from checkpoint:
```sh
python train.py data/training/oscar_v1.12_5k --checkpoint data\output\oscar_v1.12_5k\trocr-large-handwritten\e20_lr1e-06_b4_1222\best_checkpoint.pt --epochs 20
```

To use different model
```sh
python train.py data/training/oscar_v1.12_5k --epochs 5 --model naver-clova-ix/donut-base
```

## Arguments for `train.py`

Below is a detailed explanation of all arguments available in the `train.py` script:

### Positional Arguments
- **`data`** *(str)*:  
  Path to the directory containing labeled image data for training. This is a required argument.

### Optional Arguments
- **`--evaldata`** *(str)*:  
  Path to the directory containing labeled image data for evaluation.  
  Default: `data/training/balcesu_test`.

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
  python train.py data/training/oscar_v1.12_5k --epochs 5

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

Requires python 3.6 or lower, trdg package is not updated.

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

## 👤 Acknowledgments:

### Image generation:
- https://github.com/Belval/TextRecognitionDataGenerator
### Training / finetuning:
Output format designed for EasyOCR, as explained here:
- https://github.com/clovaai/deep-text-recognition-benchmark

Alternative labeling schemes possible by using: --create_csv and adapting code for respective columns.
### General:
- https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/

