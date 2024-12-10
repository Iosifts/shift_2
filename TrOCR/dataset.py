import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
import torch

class IAMDataset(Dataset):
    """
    Custom OCR dataset for IAM handwritten text recognition.
    Includes data downloading, processing, and train-test splitting.
    """
    def __init__(self, root_dir, processor, max_target_length=128, dataset_url=None, labels_file="labels.txt", fraction=1.0, split="train"):
        """
        Initialize the IAM dataset.
        
        Args:
        root_dir (str): Path to the root directory of the dataset.
        processor: A processor for image and text (e.g., TrOCRProcessor).
        max_target_length (int): Maximum length of tokenized text.
        dataset_url (str, optional): URL to download the dataset.
        labels_file (str): Name of the labels file in the dataset.
        fraction (float): Fraction of the dataset to use (e.g., 0.01 for 1%).
        split (str): Whether to use "train" or "test" split.
        """
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.labels_file = os.path.join(root_dir, labels_file)
        self.dataset_url = dataset_url
        self.fraction = fraction
        self.split = split

        # Ensure root directory exists
        os.makedirs(self.root_dir, exist_ok=True)

        # Download and preprocess the dataset if needed
        if dataset_url and not os.path.exists(self.labels_file):
            self._download_and_preprocess(dataset_url)

        # Load and split the dataset
        self._prepare_dataset()

    def _download_and_preprocess(self, url):
        """
        Downloads and preprocesses the dataset from the given URL.
        """
        txt_file = os.path.join(self.root_dir, "gt_test.txt")
        response = requests.get(url)
        response.raise_for_status()
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Preprocess the dataset
        df = pd.read_fwf(txt_file, header=None, names=["file_name", "text"])
        df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
        df.to_csv(self.labels_file, index=False)

    def _prepare_dataset(self):
        """
        Loads the dataset and splits it into train and test sets.
        """
        df = pd.read_csv(self.labels_file)

        # Sample the dataset fraction
        df = df.sample(frac=self.fraction, random_state=42)
        
        # Split into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        # Assign the appropriate split
        self.df = train_df if self.split == "train" else test_df
        self.df['file_path'] = self.df['file_name'].apply(lambda x: os.path.join(self.root_dir, "images", x))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the file name and corresponding text
        file_path = self.df['file_path'].iloc[idx]
        text = self.df['text'].iloc[idx]
        
        # Load and process the image
        image = Image.open(file_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Encode the text into labels
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # Replace pad_token_id with -100 to ignore them in the loss computation
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 
            for label in labels
        ]
        
        # Prepare the output encoding
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
        return encoding

class OCRDataset(Dataset):
    """
    Custom OCR dataset that assumes data in the following format:
    ---------------------
    dataroot
    | images
      | image0.png
      | image1.png
    | labels.txt
    ---------------------
    labels.txt contains:
        'image0.png'\t'<text label>'
        'image1.png'\t'<text label>'
        ...
    """
    def __init__(self, root_dir, processor, labels_file="labels.txt",
                 max_target_length=128, df=None):
        """
        Initialize the OCR dataset with preprocessing.

        Args:
            root_dir (str): Root directory containing the dataset.
            processor: A processor for image and text (e.g., TrOCRProcessor).
            labels_file (str): Name of the labels file in the dataset.
            max_target_length (int): Maximum length of tokenized text.
            df (pd.DataFrame, optional): If provided, use this dataframe directly 
                                         instead of reading and splitting.
        """
        self.root_dir = root_dir
        self.processor = processor
        self.labels_file = os.path.join(root_dir, labels_file)
        self.max_target_length = max_target_length

        os.makedirs(self.root_dir, exist_ok=True)

        if df is not None:
            # Use the provided DataFrame directly
            self.df = df
        else:
            # If no dataframe is provided, load from file
            df = pd.read_csv(self.labels_file, sep='\t', header=None, names=["file_name", "text"])
            df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
            df['file_path'] = df['file_name'].apply(lambda x: os.path.join(self.root_dir, "images", x))
            self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file_path'].iloc[idx]
        text = self.df['text'].iloc[idx]

        # Load and process the image
        image = Image.open(file_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Encode the text into labels
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        # Replace pad_token_id with -100 to ignore them in the loss computation
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
        return encoding

    
