<div align="center">

# Transformer-Based OCR

**A Framework for Training Transformer-Based Text Detection Models**

[Key Features](#-key-features) •
[Quick Start](#️-quick-start) •
[Supported Datasets](#-supported-datasets) •
[Training Examples](#-training-examples) •
[Inference](#-inference) •
[Data Generation](#-data-generation)

<!-- <p align="center">
  <img src="data/logo.png" alt="Project Logo" width="600"/>
</p> -->

<!-- [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) -->

</div>

## 🛠️ Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train using default settings
python train.py --epochs 5 --dataset MNIST --fraction 0.01

# Run inference
python inference.py --image_path input.png --model_path data/output/MNIST/.../best_checkpoint_char_acc.pt
```

## 📂 Supported Datasets
1. **IAM Words Dataset** - [Download Link](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)
2. **Synthetic Dataset** - Romanian examples at [Download Link](https://drive.google.com/drive/folders/1ErvjszLBqVIrO7wnsVUc6zWv5CtPmgF_?usp=sharing) or create your own using the `generator.py` script. An alternative is to use the trdg library to generate synthetic data [Download Link](https://github.com/Belval/TextRecognitionDataGenerator), they use the same dataset structure.
3. **MNIST Dataset** - (Auto-downloads)

## 🧪 Training

```sh
# Basic training
python train.py --data data/datasets/ocr_dataset --epochs 5

# Different datasets
python train.py --data data/datasets/iam_words --epochs 5 --dataset IAM
python train.py --epochs 5 --dataset MNIST --fraction 0.01  # Auto-downloads

# Alternative models
python train.py --data data/datasets/ocr_dataset --model naver-clova-ix/donut-base --batchsize 1
python train.py --data data/datasets/ocr_dataset --model facebook/nougat-base --batchsize 4

# Resume from checkpoint
python train.py --data data/datasets/ocr_dataset \
                --checkpoint data/output/ocr_dataset/trocr-large-handwritten/e20_lr1e-06_b4_1222/best_checkpoint.pt \
                --epochs 20

# Use config file
python train.py --config data/configs/ocr_config.yaml
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `data` | Training data directory path | Required |
| `--evaldata` | Evaluation data directory | `data/datasets/balcesu_test` |
| `--output` | Output directory for checkpoints/logs | `data/output` |
| `--checkpoint` | Path to resume training from | `None` |
| `--model` | Model choice (trocr-base/large, handwritten) | `microsoft/trocr-large-handwritten` |
| `--dataset` | Dataset type (IAM/MNIST/custom) | `custom` |
| `--epochs` | Number of training epochs | `5` |
| `--batchsize` | Training batch size | `4` |
| `--lr` | Learning rate | `1e-6` |
| `--lr_patience` | Epochs before LR reduction | `2` |
| `--val_iters` | Evaluation frequency (epochs) | `1` |
| `--num_samples` | Validation samples to display | `10` |

## 🔍 Inference

```bash
# Basic usage with just an image
python inference.py --image_path path/to/image.jpg

# Using a specific model and checkpoint
python inference.py --image_path path/to/image.jpg --model_path path/to/model --checkpoint_path path/to/checkpoint.pt

# Using all options
python inference.py --image_path path/to/image.jpg \
                   --model_path custom/model/path \
                   --checkpoint_path custom/checkpoint.pt \
                   --reference_text_path custom/reference.txt \
                   --predictions_file custom/predictions.json \
                   --draw # Draw bounding boxes on the image

# One-line string:
python inference.py --image_path data\inference\Balcescu.png --checkpoint_path data\output\ocr_dataset\trocr-large-handwritten\e20_lr1e-06_b4_fr1.0_tfr1.0_balcesu_test\trocr-large-handwritten-best-char_acc.pt --predictions_file data\output\ocr_dataset\trocr-large-handwritten\e20_lr1e-06_b4_fr1.0_tfr1.0_balcesu_test\trocr-large-handwritten-best-char_acc.pt
```

## 📦 Data Generation

The `generator.py` script creates synthetic OCR training data. It requires a text dataset (.txt file) and a set of fonts and backgrounds. An alternative is to use the trdg library to generate synthetic data [Download Link](https://github.com/Belval/TextRecognitionDataGenerator), they use the same dataset structure. You may run into some issues setting it up, because the trdg package does not receive support anymore. But once set up the integrated generator has better augmentation than the generator offered here.

You can download two romanian language datasets (wiki-dump and OSCAR dataset) using this script:
```sh
python data/generation/download_text_dataset.py
```

Next, consider the following input structure:
```sh
generation/
├── assets
    ├── fonts # place your fonts here
    ├── backgrounds # place your backgrounds here
└── data/
    ├── input/
        ├── text-dataset.txt # arbitrary text dataset
    ├── output/
        # empty directory
├── generator.py
```

Run this command in your environment to generate synthetic dataset:
```sh
python generator.py data/generation/data/input/text-dataset.txt data/generation/data/output/ocr_dataset --image_count 50 --sentences_per_page 1 --max_length 100 --characters "ăâîșțĂÂÎȘȚ"
```
We get a new directoru in under *generation/data/output*:
```sh
├── output/
    ├── ocr_dataset/
      ├── labels.csv
      ├── labels.txt
      └── images/
          ├── image0.png
          ├── image1.png
          └── ...
```

labels.txt:
```sh
{image0.png}\t{text0}
{image1.png}\t{text1}
...
```

You can use the generated dataset for training your OCR model by chosing the `--data` argument to point to the generated dataset and `--dataset` to `custom`.

## 👤 Acknowledgments

- https://github.com/microsoft/unilm/tree/master/trocr
- https://github.com/Belval/TextRecognitionDataGenerator
- https://github.com/naver-ai/donut
- https://github.com/facebookresearch/nougat