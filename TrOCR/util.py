from evaluate import load as load_metric
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from io import BytesIO

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

def process_data(file_path, image_dir, fraction=1.0, split=True, test_size=0.2):
    """
    Reads, processes, and optionally splits the dataset.
    """
    def normalize_apostrophes(text):
        text = text.replace("’", "'").replace("`", "'").replace("\u2019", "'")
        text = text.replace("\u00A0", " ").strip()  # Replace non-breaking spaces with standard spaces
        return text
    
    # Read data
    df = pd.read_csv(file_path, encoding="utf-8", sep='\t', header=None, names=["file_name", "text"])
    
    # Normalize apostrophes in the text column
    df['text'] = df['text'].apply(normalize_apostrophes)
    
    # File extensions and add paths
    df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
    df['file_path'] = df['file_name'].apply(lambda x: os.path.join(image_dir, x))
    
    # Sample fraction
    df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)
    
    if split:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    return None, df.reset_index(drop=True)


def create_dataset(dataframe, root_dir, processor, dataset_class):
    return dataset_class(root_dir=root_dir, processor=processor, df=dataframe)

def pil_image_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

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

