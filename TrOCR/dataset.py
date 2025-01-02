import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms

class OCRDataset(Dataset):
    """
    Dataset as created with the generator.py script.
    Examples can be found here: 
    https://drive.google.com/drive/folders/1ErvjszLBqVIrO7wnsVUc6zWv5CtPmgF_?usp=drive_link
    ---------------------
    Assumes data in the following format:
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
            # Ensure labels file exists
            if not self.labels_file.exists():
                raise FileNotFoundError(f"Labels file '{self.labels_file}' not found.")
            # If no dataframe is provided, load from file
            df = pd.read_csv(self.labels_file, sep='\t', header=None, names=["file_name", "text"])
            df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
            df['file_path'] = df['file_name'].apply(lambda x: os.path.join(self.root_dir, "images", x))
            self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = Path(self.df['file_path'].iloc[idx])
        text = self.df['text'].iloc[idx]
        if not file_path.exists():
            raise FileNotFoundError(f"Image file '{file_path}' not found.")

        image = Image.open(file_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(
            text_target=text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 
            for label in labels
        ]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
    
    @classmethod
    def create_dataset(cls, datapath, processor, fraction, test_frac, args, logger):
        """Create dataset from custom format with labels.txt file"""
        train_file = os.path.join(datapath, "labels.txt")
        df = pd.read_csv(train_file, sep='\t', header=None, names=["file_name", "text"])
        df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
        df['file_path'] = df['file_name'].apply(lambda x: os.path.join(datapath, "images", x))

        df = df.sample(frac=fraction, random_state=args.seed).reset_index(drop=True)
        train_df, remainder_df = train_test_split(df, test_size=0.4, random_state=args.seed)
        eval_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=args.seed)
        datasets = {
            'train': OCRDataset(root_dir=datapath, processor=processor, df=train_df),
            'eval': OCRDataset(root_dir=datapath, processor=processor, df=eval_df),
            'test': OCRDataset(root_dir=datapath, processor=processor, df=test_df)
        }
        
        if args.testdata:
            logger.info(f"Replacing test set with data from {args.testdata}")
            new_test_df = pd.read_csv(
                os.path.join(args.testdata, "labels.txt"), 
                sep='\t', header=None, 
                names=["file_name", "text"]
            )
            new_test_df['file_name'] = new_test_df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
            new_test_df['file_path'] = new_test_df['file_name'].apply(
                lambda x: os.path.join(args.testdata, "images", x)
            )
            new_test_df = new_test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
            datasets['test'] = OCRDataset(root_dir=args.testdata, processor=processor, df=new_test_df)
        
        return datasets
    
    @staticmethod
    def _fix_file_extension(file_name):
        """
        Fix common file extension issues.
        """
        if file_name.endswith('jp'):  # Handle cases like `.jp` files
            return file_name + 'g'
        return file_name

class IAMDataset(Dataset):
    """
    Dataset from: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database 
    Custom OCR dataset that assumes data in the following format:
    ---------------------
    dataroot
    | words
    |  | a01
    |  |  | a01-000u
    |  |  |  | a01-000u-00-00.png
    |  |  |  | a01-000u-00-01.png
    |  |  |  | a01-000u-00-02.png
    | words.txt
    ---------------------
    """
    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        max_target_length=128
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame already containing columns 
                               'file_path' and 'transcription'.
            processor: A processor for image and text (e.g. TrOCRProcessor).
            max_target_length (int): Maximum length of tokenized text.
        """
        
        self.processor = processor
        self.max_target_length = max_target_length

        # Filter out any invalid or too-small images
        # i.e. 1x1 pixel images exist in IAM
        valid_rows = []
        for idx, row in df.iterrows():
            image_path = row['file_path']
            try:
                # Attempt to open and convert to RGB
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    arr = np.array(img)
                    
                    # Check for non-trivial dimensions
                    if arr.shape[0] > 1 and arr.shape[1] > 1:
                        valid_rows.append(row)
                    else:
                        print(f"[WARNING] Skipped tiny image '{image_path}' with shape {arr.shape}")
            except Exception as e:
                print(f"[ERROR] Could not open or convert '{image_path}'. Skipping. ({e})")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[INFO] Kept {len(self.df)}/{len(df)} images after filtering.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['file_path']
        text = str(self.df.iloc[idx]['transcription'])

        image = Image.open(image_path).convert("RGB")
        encoding = self.processor(
            image,
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        )
        # Remove the batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(dim=0)
            
        return encoding
    
    @classmethod
    def create_dataset(cls, datapath, processor, fraction, test_frac, args, logger):
        """Create dataset from IAM format"""             
        labels_file = os.path.join(datapath, "words.txt")
        try:
            df = pd.read_csv(labels_file, sep='\s+', header=None, comment='#',
                names=["word_id", "segmentation_status", "graylevel", "x", "y", 
                    "width", "height", "grammatical_tag", "transcription"],
                on_bad_lines='skip', engine='python')
            logger.info(f"Dataset loaded successfully with {len(df)} entries.")
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            raise

        df['file_path'] = df['word_id'].apply(
            lambda x: os.path.join(datapath, "words",
                x.split('-')[0],                # "a01"
                x.split('-')[0] + "-" + x.split('-')[1],  # "a01-000u"
                x + ".png"                      # "a01-000u-00-00.png"
            )
        )
        df = validate_files(df, logger)
        df = df.sample(frac=fraction, random_state=args.seed).reset_index(drop=True)
        train_df, remainder_df = train_test_split(df, test_size=0.4, random_state=args.seed)
        eval_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=args.seed)
        
        datasets = {
            'train': IAMDataset(df=train_df, processor=processor, max_target_length=128),
            'eval': IAMDataset(df=eval_df, processor=processor, max_target_length=128),
            'test': IAMDataset(df=test_df, processor=processor, max_target_length=128)
        }
        
        if args.testdata:
            logger.info(f"Replacing test set with data from {args.testdata}")
            new_test_df = pd.read_csv(
                os.path.join(args.testdata, "labels.txt"), 
                sep='\t', header=None, 
                names=["file_name", "text"]
            )
            new_test_df['file_name'] = new_test_df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
            new_test_df['file_path'] = new_test_df['file_name'].apply(
                lambda x: os.path.join(args.testdata, "images", x)
            )
            # Rename 'text' column to match IAM dataset's 'transcription' column
            new_test_df = new_test_df.rename(columns={'text': 'transcription'})
            new_test_df = new_test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
            datasets['test'] = IAMDataset(df=new_test_df, processor=processor, max_target_length=128)
        
        return datasets

class MNISTDataset(Dataset):
    """
    MNIST Dataset for OCR training.
    Download and preprocessing is handled automatically.
    """
    def __init__(self, root_dir, processor, df=None, max_target_length=128):
        from torchvision.datasets import MNIST
        from torchvision import transforms
        
        self.processor = processor
        self.max_target_length = max_target_length
        self.df = df
        
        # Define transforms to convert to RGB (MNIST is grayscale)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.ToPILImage()  # Convert back to PIL for processor
        ])
        
        self.mnist = MNIST(root_dir, train=True, download=True)

    def __len__(self):
        return len(self.df) if self.df is not None else len(self.mnist)

    def __getitem__(self, idx):
        if self.df is not None:
            image_id = self.df.iloc[idx]['image_id']
            label = self.df.iloc[idx]['transcription']
            image, _ = self.mnist[image_id]
        else:
            image, label = self.mnist[idx]
            label = str(label)
        
        # Convert to RGB and preprocess
        image = self.transform(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Tokenize label
        labels = self.processor.tokenizer(
            text_target=label,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # Replace padding token id with -100 for loss calculation
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 
            for label in labels
        ]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

    @classmethod
    def create_dataset(cls, datapath, processor, fraction, test_frac, args, logger):
        """Create MNIST dataset splits"""
        
        # Load the full dataset
        mnist = MNIST(datapath, train=True, download=True)
        
        # Create a DataFrame similar to other datasets
        df = pd.DataFrame({
            'image_id': range(len(mnist)),
            'transcription': [str(label) for _, label in mnist],
        })
        
        # Split dataset
        df = df.sample(frac=fraction, random_state=args.seed).reset_index(drop=True)
        train_df, remainder_df = train_test_split(df, test_size=0.4, random_state=args.seed)
        eval_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=args.seed)
        
        datasets = {
            'train': MNISTDataset(datapath, processor, df=train_df),
            'eval': MNISTDataset(datapath, processor, df=eval_df),
            'test': MNISTDataset(datapath, processor, df=test_df)
        }
        
        if args.testdata:
            logger.info(f"Replacing test set with data from {args.testdata}")
            new_test_df = pd.read_csv(
                os.path.join(args.testdata, "labels.txt"), 
                sep='\t', 
                header=None, 
                names=["file_name", "text"]
            )
            new_test_df = new_test_df.rename(columns={'text': 'transcription'})
            new_test_df = new_test_df.sample(frac=test_frac, random_state=args.seed).reset_index(drop=True)
            datasets['test'] = MNISTDataset(args.testdata, processor, df=new_test_df)
        
        return datasets

def validate_files(df, logger):
    """Validate that all image files exist"""
    missing_files = df[~df['file_path'].apply(os.path.exists)]
    if not missing_files.empty:
        logger.warning(f"Missing {len(missing_files)} files. Example: {missing_files.iloc[0]['file_path']}")
    df = df[df['file_path'].apply(os.path.exists)]
    if df.empty:
        logger.error("No valid image files found in dataset.")
        raise ValueError("Empty dataset after validation")
    return df


def create_dataset(args, processor, fraction, test_frac, logger):
    """Factory function to create appropriate dataset"""
    dataset_classes = {
        'custom': OCRDataset,
        'IAM': IAMDataset,
        'MNIST': MNISTDataset
    }
    
    if args.dataset not in dataset_classes:
        raise ValueError(f"Dataset {args.dataset} not available")
    
    dataset_class = dataset_classes[args.dataset]
    return dataset_class.create_dataset(args.data, processor, fraction, test_frac, args, logger)

