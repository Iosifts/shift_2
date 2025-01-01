from evaluate import load as load_metric
import os
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from io import BytesIO
import matplotlib.pyplot as plt
from collections import defaultdict
import yaml
from typing import Dict, Any
import random
import numpy as np
import torch

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

def compute_metrics(pred_str, label_str):
    """Remove"""
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    # Exact match accuracy
    # Accuracy = number of exact matches / total
    exact_matches = sum([1 for p, l in zip(pred_str, label_str) if p.strip() == l.strip()])
    accuracy = exact_matches / len(label_str)

    # Character-level accuracy
    # Compute for each pair: #correct_chars / length_of_reference
    char_accuracies = []
    for p, l in zip(pred_str, label_str):
        # Compute character-level accuracy
        # We consider only up to length of the shorter string for matches
        # Or we can consider a more strict definition: 
        # number_of_matching_characters / length_of_reference
        matches = sum(pc == lc for pc, lc in zip(p, l))
        if len(l) > 0:
            char_acc = matches / len(l)
        else:
            char_acc = 1.0 if len(p) == 0 else 0.0
        char_accuracies.append(char_acc)
    char_accuracy = sum(char_accuracies) / len(char_accuracies)

    # Levenshtein distance
    # Average Levenshtein distance over all samples
    lev_dists = [levenshtein_distance(p, l) for p, l in zip(pred_str, label_str)]
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
    
def custom_decode(token_ids, tokenizer, token_to_char_map, logger):
        decoded_chars = []
        for token_id in token_ids:
            # Skip padding and special tokens explicitly
            if token_id in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                continue
            # Use manual mapping if available
            if token_id in token_to_char_map:
                decoded_chars.append(token_to_char_map[token_id])
            else:
                # Log unmapped tokens
                # logger.warning(f"Failed to decode token ID {token_id} using fallback decode.")
                decoded_char = tokenizer.decode([token_id], skip_special_tokens=False)
                decoded_chars.append(decoded_char if decoded_char != "�" else f"[UNK-{token_id}]")
        return ''.join(decoded_chars)

def dynamic_decode(token_ids, tokenizer, token_to_char_map, 
    logger, missing_tokens_freq, fallback_map=None
):
    if fallback_map is None:
        fallback_map = {}

    decoded_chars = []
    for token_id in token_ids:
        # Skip special/padding tokens
        if token_id in [
            tokenizer.pad_token_id, 
            tokenizer.bos_token_id, 
            tokenizer.eos_token_id
        ]:
            continue

        if token_id in token_to_char_map:
            # We have a direct mapping
            decoded_chars.append(token_to_char_map[token_id])
        else:
            # Record that we missed this token
            missing_tokens_freq[token_id] += 1

            # Attempt a fallback decode (if you have a fallback map or partial logic)
            decoded_str = tokenizer.decode([token_id], skip_special_tokens=False)
            if decoded_str == "�":  # Replacement char
                decoded_chars.append(f"[UNK-{token_id}]")
            else:
                # Optionally see if fallback_map can help
                # e.g. if decoded_str in fallback_map -> use fallback_map[decoded_str]
                replaced_str = decoded_str
                for k, v in fallback_map.items():
                    replaced_str = replaced_str.replace(k, v)

                # If replaced_str is non-empty, use it
                if replaced_str.strip():
                    decoded_chars.append(replaced_str)
                else:
                    decoded_chars.append(f"[UNK-{token_id}]")

    return ''.join(decoded_chars)


def create_unique_directory(base_dir, dir_name):
    full_path = os.path.join(base_dir, dir_name)
    counter = 1
    while os.path.exists(full_path):
        full_path = os.path.join(base_dir, f"{dir_name}_{counter}")
        counter += 1
    os.makedirs(full_path, exist_ok=True)
    return full_path

def plot_metric(
    train_data: list[tuple[int, float]], 
    val_data: list[tuple[int, float]],
    test_data: list[tuple[int, float]],
    metric_name: str, 
    output_dir: str, 
    logger,
    show_epoch: bool = False,
    iters_per_epoch: int = None,
    use_log_scale: bool = False,
    high_value_threshold: float = 10.0
):
    """
    train_data / val_data / test_data: Lists of (iteration_step, metric_value)
      e.g. [(0, 11.2), (1, 10.5), (2, 9.8), ...]
      
    metric_name: e.g. "loss", "wer", "char_acc"
    output_dir: directory to save the plot (pdf)
    logger: your logging instance
    
    show_epoch: if True, label & tick x-axis in epochs instead of raw iterations
    iters_per_epoch: integer # of iterations (batches) per epoch (e.g. 121).
                     Used only if show_epoch=True.
    """
    # If everything is empty, do nothing.
    if not train_data and not val_data and not test_data:
        return

    plt.figure(figsize=(10, 6))

    # -------------------------
    #  PLOT THE LINES
    # -------------------------
    # Plot TRAIN
    if train_data:
        x_steps_train, y_values_train = zip(*train_data)
        plt.plot(x_steps_train, y_values_train, label=f"Train {metric_name}", color='blue')
    # Plot VAL
    if val_data:
        x_steps_val, y_values_val = zip(*val_data)
        plt.plot(x_steps_val, y_values_val, label=f"Val {metric_name}", color='orange')
    # Plot TEST
    if test_data:
        x_steps_test, y_values_test = zip(*test_data)
        plt.plot(x_steps_test, y_values_test, label=f"Test {metric_name}", color='green')

    # -------------------------
    #  Automatically decide on log scale?
    # -------------------------
    if use_log_scale:
        # Force log scale
        plt.yscale('log')
    else:
        # Or detect large values automatically
        all_values = []
        if train_data:
            all_values.extend(v for _, v in train_data)
        if val_data:
            all_values.extend(v for _, v in val_data)
        if test_data:
            all_values.extend(v for _, v in test_data)

        if all_values and max(all_values) > high_value_threshold:
            plt.yscale('log')

    # -------------------------
    #  EPOCH MODE vs. ITER MODE
    # -------------------------
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
        plt.xlabel("Iteration")
        plt.title(f"{metric_name} vs. Iteration")

    # -------------------------
    #  FINALIZE & SAVE
    # -------------------------
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)

    # Decide naming convention for file
    x_axis_label = "epoch" if show_epoch else "iteration"
    save_name = f"{metric_name}_train_val_test_{x_axis_label}.pdf"
    save_path = os.path.join(output_dir, save_name)

    plt.savefig(save_path)
    logger.info(f"Saved {metric_name} plot (Train, Val, Test) to {save_path}")
    plt.close()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict[Any, Any]:
    """Load YAML config file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)