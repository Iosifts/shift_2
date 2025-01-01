import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
from datetime import datetime
from transformers import TrOCRProcessor
from transformers import logging
from transformers import VisionEncoderDecoderModel
from transformers import DonutProcessor, AutoImageProcessor, AutoTokenizer
logging.set_verbosity_error()  # Only errors will be printed
from transformers import GenerationConfig
from transformers import AutoModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import logging
from logging import FileHandler, StreamHandler, Formatter
import util
from util import custom_decode, plot_metric, set_seed, load_config
import models
from models import get_processor_and_model, train_step, generate_step, get_model_specific_config
import dataset
from dataset import OCRDataset, IAMDataset
import argparse
import random
import numpy as np
from collections import defaultdict
import yaml
from typing import Dict, Any
missing_tokens_freq = defaultdict(int)

def parse_args():
    parser = argparse.ArgumentParser('TrOCR Training')
    # Overwrite following args with config.yaml
    parser.add_argument('--config', type=str, default=None,
        help='Path to YAML config file that overrides command line arguments')
    parser.add_argument('--data', type=str, 
        help='Directory containing labeled image data')
    parser.add_argument('--testdata', type=str, default='data/datasets/balcesu_test',
        help='Directory containing labeled image data to replace test split distribution')
    parser.add_argument('--output', type=str, default='data/output', 
        help='Directory in which to store checkpoints, logs, etc.')
    parser.add_argument('--checkpoint', type=str, default=None, 
        help='Path to checkpoint (.pt currently)')
    parser.add_argument('--model', default='microsoft/trocr-large-handwritten', choices=[
            # TrOCR models
            'microsoft/trocr-base-stage1', 
            'microsoft/trocr-large-stage1',
            'microsoft/trocr-base-handwritten', 
            'microsoft/trocr-large-handwritten',
            'microsoft/trocr-small-handwritten',
            'microsoft/trocr-base-printed',
            'microsoft/trocr-small-printed',
            # Donut models
            'naver-clova-ix/donut-base',
            'naver-clova-ix/donut-base-finetuned-rvlcdip',
            'naver-clova-ix/donut-proto',
            # ViT-based models
            'facebook/nougat-base',
            'facebook/nougat-small',
            'microsoft/dit-base',
            'microsoft/dit-large',
            # Custom models (TODO)
            'custom'
        ], 
        help='Select the model to use.')
    parser.add_argument('--dataset', default='custom', choices=[
            'IAM', 
            'custom'
        ],
        help='Select the dataset to use. For choice \'custom\' refer to dataset.OCRDataset')
    parser.add_argument('--epochs', type=int, default=5, 
        help='Epochs to train')
    parser.add_argument('--batchsize', type=int, default=4, 
        help='Batchsize of DataLoader')
    parser.add_argument('--val_iters', type=int, default=1, 
        help='Number of epochs to eval at')
    parser.add_argument('--test_iters', type=int, default=2,
        help='Run test loop every N epochs')
    parser.add_argument('--lr', type=float, default=1e-6, 
        help='Learning rate of update step')
    parser.add_argument('--lr_patience', type=int, default=2, 
        help='Patience for lr scheduler')
    parser.add_argument('--num_samples', type=float, default=10, 
        help='Number of printed sample predictions')
    parser.add_argument('--seed', type=int, default=42, 
        help='Patience for lr scheduler')
    parser.add_argument('--special_chars', nargs='+', default=["ă", "â", "î", "ș", "ț", "Ă", "Â", "Î", "Ș", "Ț"],
        help='List of special characters to add to tokenizer')
    
    args = parser.parse_args()

    if args.config is not None:
        config = load_config(args.config)
        args_dict = vars(args)
        for key, value in config.items():
            if value is not None:  # Override if value is provided
                args_dict[key] = value

    return args

def create_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # os.makedirs(output_dir, exist_ok=True)
    file_handler = FileHandler(
        os.path.join(output_dir, "training.log"), 
        encoding='utf-8', errors='replace') 
    # file_handler.setLevel(logging.INFO)
    formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Initialized Logging")
    return logger

def train():
    args = parse_args()
    set_seed(args.seed)
    fraction = 0.05 # fraction of train data used 
    test_frac = 0.5
    change_test = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Extract and create paths
    datapath = args.data
    base_output_dir = args.output
    os.makedirs(base_output_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(datapath))
    model_name = os.path.basename(os.path.normpath(args.model))
    nested_output_dir = os.path.join(base_output_dir, dataset_name, model_name)
    os.makedirs(nested_output_dir, exist_ok=True)
    # Create outdir name
    extraname = 'balcescu' if change_test else ''
    timestamp = time.strftime("%m%d")
    formatted_lr = util.format_lr(args.lr)
    output_dir_name = f"e{args.epochs}_lr{formatted_lr}" + \
        f"_b{args.batchsize}_fr{fraction}_tfr{test_frac}" + \
        f"{extraname}"
    if args.checkpoint is not None:
        output_dir_name += f"_ckpt" 
    output_dir = util.create_unique_directory(nested_output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir) # Set up logger

    """Initialize model and data processor"""
    model_config = get_model_specific_config(args)
    processor, model = get_processor_and_model(args)
    model.to(device)
    if model_config['generation_config'] is not None:
            model.generation_config = model_config['generation_config']
            logger.info(f"Updated generation config: {model.generation_config}")
    
    """Create dataset"""
    try:
        datasets = dataset.create_dataset(args, processor, fraction, test_frac, logger)
        train_dataset = datasets['train']
        eval_dataset = datasets['eval']
        test_dataset = datasets['test']
        logger.info(f"Number of training examples: {len(train_dataset)}")
        logger.info(f"Number of validation examples: {len(eval_dataset)}")
        logger.info(f"Number of test examples: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        exit(1)
    encoding = train_dataset[0]
    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.decode(labels, skip_special_tokens=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batchsize)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize)

    """Tokenization"""
    special_chars = args.special_chars
    processor.tokenizer.add_tokens(special_chars, special_tokens=False)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    char_to_token_map = {}
    token_to_char_map = {}
    for char in special_chars:
        token_id = processor.tokenizer.convert_tokens_to_ids(char)
        char_to_token_map[char] = token_id
        token_to_char_map[token_id] = char
    for char in special_chars: # Verify decoder mappings
        token_id = char_to_token_map[char]
        decoded_char = processor.tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded_char != char:
            logger.error(f"Decoder Mapping Error: '{char}' -> Token ID {token_id} -> Decoded as '{decoded_char}'")
        else:
            logger.info(f"Decoder Mapping Success: '{char}' -> Token ID {token_id} -> Decoded as '{decoded_char}'")

    """Training Setup"""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience)
    best_val_loss = float('inf')
    best_val_char_acc = -1.0
    train_metrics = {'loss': []}
    val_metrics = {'loss': [], 'char_acc': [], 'wer': []}
    test_metrics = {'loss': [], 'char_acc': [], 'wer': []}

    start_epoch = 0
    if args.checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        best_val_char_acc = checkpoint['best_val_char_acc']
        start_epoch = checkpoint['epoch']
        train_metrics = checkpoint['train_metrics']
        val_metrics = checkpoint['val_metrics']
        logger.info(f"Resumed at epoch {start_epoch} with best_val_loss {best_val_loss:.4f}")

    """Training"""
    global_step = 0
    epoch = 1

    while True:
        # -------------------- TRAIN --------------------
        if epoch <= args.epochs:
            model.train()
            running_train_loss = 0.0

            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}", dynamic_ncols=True):
                loss = train_step(model, batch, device, args.model)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                running_train_loss += loss.item()
                train_metrics['loss'].append((global_step, loss.item()))
                global_step += 1

            avg_train_loss = running_train_loss / len(train_dataloader)
            logger.info(f"Train loss after epoch {epoch}: {avg_train_loss}")
        else: 
            logger.info(f"Reached epoch {epoch}. Finishing Training.")
            break

        # -------------------- EVAL ---------------------
        if epoch % args.val_iters == 0:
            model.eval()
            total_val_loss = 0.0
            total_cer, total_wer, total_acc, total_char_acc, total_lev_dist = 0, 0, 0, 0, 0
            count = 0
            sample_preds, sample_refs, sample_confs = [], [], []

            with torch.no_grad():
                for batch in tqdm(eval_dataloader, dynamic_ncols=True):
                    val_loss = train_step(model, batch, device, args.model)
                    total_val_loss += val_loss

                    generation_outputs = generate_step(
                        model, 
                        batch, 
                        device, 
                        args.model,
                        model.generation_config
                    )

                    pred_ids = generation_outputs.sequences
                    scores_list = generation_outputs.scores
                    
                    # Get the labels from batch
                    labels = batch["labels"]
                    labels_adj = labels.clone()
                    labels_adj[labels_adj == -100] = processor.tokenizer.pad_token_id
                    
                    # Decode predictions and labels
                    label_str = processor.batch_decode(labels_adj, skip_special_tokens=True)
                    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

                    # Make sure pred_str and label_str have the same batch size
                    if len(pred_str) != len(label_str):
                        if len(pred_str) == 1:
                            pred_str = pred_str * len(label_str)  # Repeat prediction to match batch size
                        else:
                            pred_str = pred_str[:len(label_str)]  # Truncate to match batch size

                    # Compute metrics
                    cer, wer, acc, char_acc, lev_dist = util.compute_metrics(pred_str, label_str)
                    total_cer += cer
                    total_wer += wer
                    total_acc += acc
                    total_char_acc += char_acc
                    total_lev_dist += lev_dist
                    count += 1

                    """Collect samples with confidence scores"""
                    if len(sample_preds) < args.num_samples:
                        batch_size = pred_ids.size(0)
                        seq_len = pred_ids.size(1)
                        sample_needed = args.num_samples - len(sample_preds)
                        sample_size = min(batch_size, sample_needed)

                        # Compute confidence scores
                        for i in range(sample_size):
                            token_ids = pred_ids[i]
                            token_probs = []
                            for step_idx in range(seq_len - 1):
                                logits = scores_list[step_idx][i]
                                probs = torch.softmax(logits, dim=-1)
                                chosen_token_id = token_ids[step_idx+1]
                                token_prob = probs[chosen_token_id].item()
                                token_probs.append(token_prob)
                            confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
                            sample_confs.append(confidence)

                        sample_preds.extend(pred_str[:sample_size])
                        sample_refs.extend(label_str[:sample_size])

                        if len(sample_preds) >= args.num_samples:
                            pass

            """Metrics, scheduler, checkpointing"""
            avg_val_loss = total_val_loss / count if count > 0 else 0
            avg_val_cer = total_cer / count if count > 0 else 0
            avg_val_wer = total_wer / count if count > 0 else 0
            avg_val_acc = total_acc / count if count > 0 else 0
            avg_val_char_acc = total_char_acc / count if count > 0 else 0
            avg_val_lev_dist = total_lev_dist / count if count > 0 else 0

            val_metrics['loss'].append((global_step, avg_val_loss))
            val_metrics['wer'].append((global_step, avg_val_wer))
            val_metrics['char_acc'].append((global_step, avg_val_char_acc))

            scheduler.step(avg_val_loss) # step on val loss

            logger.info(
                f"[EVAL] Epoch {epoch} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"CER: {avg_val_cer:.4f} | "
                f"WER: {avg_val_wer:.4f} | "
                f"Acc: {avg_val_acc:.4f} | "
                f"CharAcc: {avg_val_char_acc:.4f} | "
                f"LevDist: {avg_val_lev_dist:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e}"
            )

            if sample_preds and sample_refs:
                logger.info("[EVAL] Sample Predictions:")
                logger.info("-" * 80)
                logger.info(f"{'Prediction':<35} | {'Reference':<35} | {'Confidence':<8}")
                logger.info("-" * 80)
                for p, r, c in zip(sample_preds, sample_refs, sample_confs):
                    logger.info(f"{p[:35]:<35} | {r[:35]:<35} | {c:.4f}")
                logger.info("-" * 80)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_char_acc': best_val_char_acc,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }, best_checkpoint_path)
                logger.info(f"New best checkpoint saved to {best_checkpoint_path} with Val Loss: {best_val_loss:.4f}")

            if avg_val_char_acc > best_val_char_acc:
                best_val_char_acc = avg_val_char_acc
                best_checkpoint_char_acc_path = os.path.join(output_dir, "best_checkpoint_char_acc.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_char_acc': best_val_char_acc,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }, best_checkpoint_char_acc_path)
                logger.info(f"New best checkpoint saved to {best_checkpoint_char_acc_path} with CharAcc: {best_val_char_acc:.4f}")

        # -------------------- TEST ---------------------
        if epoch % args.test_iters == 0:
            model.eval()
            total_test_loss  = 0.0
            total_cer, total_wer, total_acc, total_char_acc, total_lev_dist = 0, 0, 0, 0, 0 # overwrite val
            count_test  = 0
            sample_preds_test, sample_refs_test, sample_confs_test = [], [], []

            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc=f"Test Epoch {epoch}", dynamic_ncols=True):
                    test_loss = train_step(model, batch, device, args.model)
                    total_test_loss += test_loss

                    generation_outputs = generate_step(
                        model, 
                        batch, 
                        device, 
                        args.model,
                        model.generation_config
                    )
                    pred_ids = generation_outputs.sequences
                    scores_list = generation_outputs.scores

                    labels = batch["labels"]
                    labels_adj = labels.clone()
                    labels_adj[labels_adj == -100] = processor.tokenizer.pad_token_id

                    label_str = processor.batch_decode(labels_adj, skip_special_tokens=True)
                    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

                    if len(pred_str) != len(label_str):
                        if len(pred_str) == 1:
                            pred_str = pred_str * len(label_str)  # Repeat prediction to match batch size
                        else:
                            pred_str = pred_str[:len(label_str)]  # Truncate to match batch size

                    cer, wer, acc, char_acc, lev_dist = util.compute_metrics(pred_str, label_str)
                    total_cer += cer
                    total_wer += wer
                    total_acc += acc
                    total_char_acc += char_acc
                    total_lev_dist += lev_dist
                    count_test += 1

                    """Collect samples with confidence scores"""
                    if len(sample_preds_test) < args.num_samples:
                        batch_size = pred_ids.size(0)
                        seq_len = pred_ids.size(1)
                        sample_needed = args.num_samples - len(sample_preds_test)
                        sample_size = min(batch_size, sample_needed)

                        # Compute confidence scores
                        for i in range(sample_size):
                            token_ids = pred_ids[i]
                            token_probs = []
                            for step_idx in range(seq_len - 1):
                                logits = scores_list[step_idx][i]
                                probs = torch.softmax(logits, dim=-1)
                                chosen_token_id = token_ids[step_idx+1]
                                token_prob = probs[chosen_token_id].item()
                                token_probs.append(token_prob)
                            confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
                            sample_confs_test.append(confidence)

                        sample_preds_test.extend(pred_str[:sample_size])
                        sample_refs_test.extend(label_str[:sample_size])

                        if len(sample_preds_test) >= args.num_samples:
                            pass

            """Metrics"""
            avg_test_loss = total_test_loss / count_test if count_test > 0 else 0
            avg_test_cer = total_cer / count_test if count_test > 0 else 0
            avg_test_wer = total_wer / count_test if count_test > 0 else 0
            avg_test_acc = total_acc / count_test if count_test > 0 else 0
            avg_test_char_acc = total_char_acc / count_test if count_test > 0 else 0
            avg_test_lev_dist = total_lev_dist / count_test if count_test > 0 else 0
            
            test_metrics['loss'].append((global_step, avg_test_loss))
            test_metrics['wer'].append((global_step, avg_test_wer))
            test_metrics['char_acc'].append((global_step, avg_test_char_acc))

            logger.info(
                f"[TEST] Epoch {epoch} | "
                f"Test Loss: {avg_test_loss:.4f} | "
                f"CER: {avg_test_cer:.4f} | "
                f"WER: {avg_test_wer:.4f} | "
                f"Acc: {avg_test_acc:.4f} | "
                f"CharAcc: {avg_test_char_acc:.4f} | "
                f"LevDist: {avg_test_lev_dist:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e}"
            )

            if sample_preds_test and sample_refs_test:
                logger.info("[TEST] Sample Predictions:")
                logger.info("-" * 80)
                logger.info(f"{'Prediction':<35} | {'Reference':<35} | {'Confidence':<8}")
                logger.info("-" * 80)
                for p, r, c in zip(sample_preds_test, sample_refs_test, sample_confs_test):
                    logger.info(f"{p[:35]:<35} | {r[:35]:<35} | {c:.4f}")
                logger.info("-" * 80)
    
        epoch += 1
        

    """Plotting & Saving"""
    plot_metric( # Track train, val, test losses simultaneously
        train_metrics['loss'], 
        val_metrics['loss'], 
        test_metrics['loss'],
        metric_name="loss", 
        output_dir=output_dir, 
        logger=logger,
        use_log_scale=True
    )

    plot_metric( # Track test metrics
        [], [], test_metrics['wer'], metric_name="wer", output_dir=output_dir, 
        logger=logger, show_epoch=True, iters_per_epoch=len(train_dataloader))

    plot_metric(
        [], [], test_metrics['char_acc'], metric_name="char_acc", output_dir=output_dir, 
        logger=logger, show_epoch=True, iters_per_epoch=len(train_dataloader))

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train()