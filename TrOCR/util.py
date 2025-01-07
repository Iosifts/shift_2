from evaluate import load as load_metric
import os
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from collections import defaultdict
import yaml
from typing import Dict, Any
import random
import numpy as np
import torch
from dataset import (
    get_dataset
)
import util
import time
import logging
from logging import FileHandler, StreamHandler, Formatter
import json

cer_metric = load_metric("cer")
wer_metric = load_metric("wer")

def levenshtein_distance(str1, str2):
    # Standard DP implementation for Levenshtein distance
    m, n = len(str1), len(str2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1,    # deletion
                           dp[i][j-1]+1,    # insertion
                           dp[i-1][j-1]+cost) # substitution
    return dp[m][n]

def compute_metrics(pred_str, label_str, logger):
    """
    Compute metrics while handling empty strings
    """
    # Filter out empty strings and their corresponding pairs
    valid_pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip() != '']
    if not valid_pairs:
        return 0.0, 0.0, 0.0, 0.0, 0.0  # Return zeros if no valid pairs
        
    pred_str_filtered, label_str_filtered = zip(*valid_pairs)
    
    # Ensure we have at least one non-empty string
    if len(pred_str_filtered) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Replace empty predictions with space to avoid metric computation errors
    pred_str_filtered = [p if p.strip() != '' else ' ' for p in pred_str_filtered]
    
    # Compute metrics
    try:
        cer = cer_metric.compute(predictions=pred_str_filtered, references=label_str_filtered)
        wer = wer_metric.compute(predictions=pred_str_filtered, references=label_str_filtered)
    except Exception as e:
        logger.error(f"Error computing CER/WER metrics: {e}")
        logger.error(f"Sample predictions: {pred_str_filtered[:5]}")
        logger.error(f"Sample references: {label_str_filtered[:5]}")
        cer, wer = 1.0, 1.0  # Worst case when computation fails
    
    # Exact match accuracy
    exact_matches = sum([1 for p, l in zip(pred_str_filtered, label_str_filtered) 
                        if p.strip() == l.strip()])
    accuracy = exact_matches / len(label_str_filtered)

    # Character-level accuracy
    char_accuracies = []
    for p, l in zip(pred_str_filtered, label_str_filtered):
        matches = sum(pc == lc for pc, lc in zip(p, l))
        if len(l) > 0:
            char_acc = matches / len(l)
        else:
            char_acc = 1.0 if len(p) == 0 else 0.0
        char_accuracies.append(char_acc)
    char_accuracy = sum(char_accuracies) / len(char_accuracies)

    # Levenshtein distance
    lev_dists = [levenshtein_distance(p, l) for p, l in zip(pred_str_filtered, label_str_filtered)]
    avg_lev_dist = sum(lev_dists) / len(lev_dists)

    return cer, wer, accuracy, char_accuracy, avg_lev_dist

def compute_all_metrics(predicted_text, reference_text):
    """
    Compute the requested metrics on a single pair of predicted_text and reference_text:
    LDist, Char Acc, Word Acc, Norm LD, Lev Acc, CER, WER
    """
    # Compute CER and WER using the evaluate metrics
    cer = cer_metric.compute(predictions=[predicted_text], references=[reference_text])
    wer = wer_metric.compute(predictions=[predicted_text], references=[reference_text])

    # Levenshtein distance (character-level)
    ldist = levenshtein_distance(predicted_text, reference_text)

    # Reference length at character level
    ref_len = len(reference_text)

    # CER and WER are already computed
    # Char Acc: typically 1 - CER
    if ref_len > 0:
        char_acc = 1.0 - cer
    else:
        char_acc = 1.0 if len(predicted_text) == 0 else 0.0

    # Word-level calculations
    pred_words = predicted_text.split()
    ref_words = reference_text.split()
    ref_word_count = len(ref_words)

    # Word Accuracy: 1 - WER
    word_acc = 1 - wer

    # Normalized LD: LDist / max(len(predicted_text), ref_len)
    norm_ld = ldist / max(len(predicted_text), ref_len) if max(len(predicted_text), ref_len) > 0 else 0.0

    # Lev Acc: 1 - (LDist / ref_len)
    lev_acc = 1 - (ldist / ref_len) if ref_len > 0 else (1.0 if len(predicted_text) == 0 else 0.0)

    return {
        "LDist": ldist,
        "Char Acc": char_acc,
        "Word Acc": word_acc,
        "Norm LD": norm_ld,
        "Lev Acc": lev_acc,
        "CER": cer,
        "WER": wer
    }



def format_lr(lr):
    if 'e' in f"{lr}":
        parts = f"{lr}".split('e')
        return f"{parts[0]}e{parts[1]}"
    else:
        return f"{lr}".replace('.', 'p')
    
def create_unique_directory(base_dir, dir_name):
    full_path = os.path.join(base_dir, dir_name)
    counter = 1
    while os.path.exists(full_path):
        full_path = os.path.join(base_dir, f"{dir_name}_{counter}")
        counter += 1
    os.makedirs(full_path, exist_ok=True)
    return full_path

def plot_metric(train_data=None, val_data=None, test_data=None, metric_name="loss", 
                output_dir=None, logger=None, use_log_scale=False, 
                show_epoch=False, iters_per_epoch=None, high_value_threshold=10):
    """Plot training/validation/test metrics"""
    # TODO: Highlight learning rate changes in the plot. Needs checkpoint data potentially.

    plt.figure(figsize=(10, 6))

    x_values_train, y_values_train = [], []
    x_values_val, y_values_val = [], []
    x_values_test, y_values_test = [], []

    def process_data(data):
        """Helper to process data and move to CPU if needed"""
        if not data:
            return [], []
        x_steps, y_values = [], []
        
        for x, y in data:
            if torch.is_tensor(x):
                x = x.cpu().item()
            x_steps.append(float(x))
            if torch.is_tensor(y):
                y = y.cpu().item()
            y_values.append(float(y))
            
        return x_steps, y_values

    if train_data:
        x_values_train, y_values_train = process_data(train_data)
        plt.plot(x_values_train, y_values_train, label=f"Train {metric_name}", color='blue')
    
    if val_data:
        x_values_val, y_values_val = process_data(val_data)
        plt.plot(x_values_val, y_values_val, label=f"Val {metric_name}", color='orange')
    
    if test_data:
        x_values_test, y_values_test = process_data(test_data)
        plt.plot(x_values_test, y_values_test, label=f"Test {metric_name}", color='green')

    if use_log_scale:
        plt.yscale('log')
        formatter = LogFormatter(labelOnlyBase=False)
        plt.gca().yaxis.set_major_formatter(formatter)
    else:
        all_values = []
        if train_data:
            all_values.extend(v for _, v in train_data)
        if val_data:
            all_values.extend(v for _, v in val_data)
        if test_data:
            all_values.extend(v for _, v in test_data)
        if all_values and max(all_values) > high_value_threshold:
            plt.yscale('log')
            formatter = LogFormatter(labelOnlyBase=False)
            plt.gca().yaxis.set_major_formatter(formatter)

    # Set y-axis limits for better focus
    plt.ylim(min(min(y_values_train, default=0), min(y_values_val, default=0), min(y_values_test, default=0)) * 0.9,
             max(max(y_values_train, default=1), max(y_values_val, default=1), max(y_values_test, default=1)) * 1.1)
    # Improve tick formatting
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    if show_epoch and iters_per_epoch is not None and iters_per_epoch > 0:
        # We'll display the X-axis ticks in units of epochs (0,1,2,3,...)
        # but the actual data remain at raw iteration positions (0, 200, 400,...).
        
        # 1) Determine the maximum iteration among the three datasets.
        max_iter = 0
        if train_data:
            max_iter = max(max_iter, max(x for x, _ in train_data))
        if val_data:
            max_iter = max(max_iter, max(x for x, _ in val_data))
        if test_data:
            max_iter = max(max_iter, max(x for x, _ in test_data))

        # 2) Figure out how many total epochs that corresponds to (round up).
        num_epochs = math.ceil(max_iter / iters_per_epoch)

        # 3) Create tick positions = [0, iters_per_epoch, 2*iters_per_epoch, ...].
        tick_positions = [epoch_idx * iters_per_epoch for epoch_idx in range(num_epochs + 1)]
        tick_labels = list(range(num_epochs + 1))

        # 4) Apply them
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel("Epoch")
        plt.title(f"{metric_name} vs. Epoch")
    else:
        # Default iteration-based labeling
        # One iteration = one batch in training
        plt.xlabel("Iteration")
        plt.title(f"{metric_name} vs. Iteration")

    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f"{metric_name}.pdf")
    plt.savefig(save_path)
    logger.info(f"Saved {metric_name} plot to {save_path}")
    plt.close()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load YAML config file with type conversion for scientific notation."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert scientific notation strings to float
    if 'lr' in config:
        try:
            config['lr'] = float(config['lr'])
        except (ValueError, TypeError):
            print(f"Warning: Could not convert learning rate '{config['lr']}' to float")
    
    return config
    
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_metrics, 
                    train_metrics, val_metrics, test_metrics, output_dir, model_name, logger):
    """
    Save model checkpoint if current metrics beat previous best.
    
    Args:
        metrics (dict): Current metrics values
        best_metrics (dict): Best metrics values so far
        *_metrics (dict): History of metrics for plotting
    Returns:
        dict: Updated best metrics
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metrics': best_metrics,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

    # Check each metric we want to track
    metrics_to_check = {
        'loss': ('min', 'Loss'),
        'char_acc': ('max', 'CharAcc'),
        # Add more metrics here as needed
    }

    updated_best = best_metrics.copy()
    
    for metric_name, (mode, display_name) in metrics_to_check.items():
        current_value = metrics[metric_name]
        best_value = best_metrics[metric_name]
        
        is_better = (current_value < best_value if mode == 'min' 
                    else current_value > best_value)
        
        if is_better:
            updated_best[metric_name] = current_value
            checkpoint_path = os.path.join(output_dir, f"{model_name}-best-{metric_name}.pt")
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(
                f"New best checkpoint saved to {checkpoint_path} "
                f"with Test {display_name}: {current_value:.4f}"
            )
    
    return updated_best

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, logger):
    """
    Load model checkpoint and associated training state.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        device: The device to load the checkpoint on
        logger: Logger instance for output
        
    Returns:
        tuple: (epoch, best_metrics, train_metrics, val_metrics, test_metrics)
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model and training states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get training progress info
    epoch = checkpoint['epoch']
    best_metrics = checkpoint['best_metrics']
    
    # Initialize metrics containers to match train.py structure
    train_metrics = {'loss': []}
    val_metrics = {'loss': [], 'char_acc': [], 'wer': []}
    test_metrics = {'loss': [], 'char_acc': [], 'wer': []}
    
    # Load metrics from checkpoint
    if 'train_metrics' in checkpoint:
        train_metrics = checkpoint['train_metrics']
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
    if 'test_metrics' in checkpoint:
        test_metrics = checkpoint['test_metrics']
    
    logger.info(f"Resumed at epoch {epoch} with best metrics: {best_metrics}")
    
    return epoch, best_metrics, train_metrics, val_metrics, test_metrics

def setup_directories(args):
    """Set up directory structure for training outputs"""
    if args.data is None:
        dataset_class = get_dataset(args.dataset)
        if dataset_class and hasattr(dataset_class, 'downloadable') and dataset_class.downloadable:
            mnist_dir = os.path.join('data', 'datasets', args.dataset)
            os.makedirs(mnist_dir, exist_ok=True)
            args.data = mnist_dir
        else:
            raise ValueError(f"No data path provided and dataset '{args.dataset}' is not downloadable")
    base_output_dir = args.output
    os.makedirs(base_output_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.data))
    model_name = os.path.basename(os.path.normpath(args.model))
    nested_output_dir = os.path.join(base_output_dir, dataset_name, model_name)
    os.makedirs(nested_output_dir, exist_ok=True)

    total_epochs = args.epochs
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        total_epochs = checkpoint['epoch'] + args.epochs
    dir_components = {
        'epochs': f"e{total_epochs}",
        'lr': f"lr{util.format_lr(args.lr)}",
        'batch': f"b{args.batchsize}",
        'fraction': f"fr{args.fraction}",
        'test_frac': f"tfr{args.test_frac}",
        'test_name': os.path.basename(os.path.normpath(args.testdata)) if args.testdata else '',
        'checkpoint': 'ckpt' if args.checkpoint else '',
        'timestamp': time.strftime("%m%d") if args.timestamp else '',
        'suffix': args.suffix if args.suffix else '',
    }
    output_dir_name = '_'.join(filter(None, [
        dir_components['epochs'],
        dir_components['lr'],
        dir_components['batch'],
        dir_components['fraction'],
        dir_components['test_frac'],
        dir_components['test_name'],
        dir_components['checkpoint'],
        dir_components['suffix']
    ]))
    output_dir = util.create_unique_directory(nested_output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    return model_name, output_dir, predictions_dir

def create_logger(output_dir):
    """Creates and configures a logger with both console and file handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = StreamHandler()
    file_handler = FileHandler(
        os.path.join(output_dir, "training.log"), 
        encoding='utf-8', errors='replace') 
    formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Initialized Logging")
    return logger

def update_tokenizer(processor, model, special_chars, logger):
    """Updates the tokenizer with special characters and verifies the mappings."""
    
    # Check which special characters need to be added
    existing_tokens = set(processor.tokenizer.get_vocab().keys())
    chars_to_add = [char for char in special_chars if char not in existing_tokens]
    
    if not chars_to_add:
        logger.info("All special characters already exist in tokenizer vocabulary")
        # Create mapping for existing special characters
        char_to_token_map = {char: processor.tokenizer.convert_tokens_to_ids(char) 
                           for char in special_chars}
    else:
        # Add new special characters to tokenizer
        num_added_tokens = processor.tokenizer.add_tokens(chars_to_add, special_tokens=True)
        logger.info(f"Added {num_added_tokens} new special tokens to tokenizer")
        
        # Resize model embeddings
        if hasattr(model, 'decoder'):
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
        elif hasattr(model, 'text_decoder'):
            old_embeddings = model.text_decoder.embedding
            new_embeddings = torch.nn.Embedding(
                len(processor.tokenizer), 
                old_embeddings.embedding_dim,
                padding_idx=processor.tokenizer.pad_token_id
            )
            new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
            model.text_decoder.embedding = new_embeddings
        
        # Create mapping for all special characters
        char_to_token_map = {char: processor.tokenizer.convert_tokens_to_ids(char) 
                           for char in special_chars}
    
    # Create reverse mapping
    token_to_char_map = {v: k for k, v in char_to_token_map.items()}
    
    # Verify mappings
    logger.info("\nVerifying character mappings:")
    for char, token_id in char_to_token_map.items():
        logger.info(f"Mapped '{char}' to token ID {token_id}")
        
    return char_to_token_map, token_to_char_map

def custom_decode(token_ids, tokenizer, token_to_char_map, logger):
    """Custom decoder that uses manual mappings for special characters"""
    decoded_chars = []
    for token_id in token_ids:
        # Skip padding and special tokens explicitly
        if token_id in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
            continue
        
        # Use manual mapping if available
        if token_id in token_to_char_map:
            decoded_chars.append(token_to_char_map[token_id])
        else:
            # Use regular tokenizer for other tokens
            try:
                char = tokenizer.decode([token_id], skip_special_tokens=True)
                if char:  # Only append if we got something back
                    decoded_chars.append(char)
            except Exception as e:
                logger.warning(f"Failed to decode token {token_id}: {e}")
                continue
            
    return ''.join(decoded_chars)

def compute_json_metrics(predictions_file):
    """
    Compute metrics from a predictions JSON file containing multiple prediction-reference pairs.
    
    Args:
        predictions_file (str): Path to the JSON file containing predictions
        
    Returns:
        dict: Dictionary containing aggregated metrics and their mean values
    """
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize lists to store metrics for each prediction
    all_metrics = []
    
    # Compute metrics for each prediction-reference pair
    for item in data['predictions']:
        pred_text = item['prediction'].strip()
        ref_text = item['reference'].strip()
        
        # Compute metrics for this pair
        pair_metrics = compute_all_metrics(pred_text, ref_text)
        all_metrics.append(pair_metrics)
    
    # Calculate mean values for each metric
    mean_metrics = {}
    if all_metrics:
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            mean_metrics[metric_name] = sum(values) / len(values)
    
    return {
        'individual_metrics': all_metrics,
        'mean_metrics': mean_metrics
    }
