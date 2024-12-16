from evaluate import load as load_metric
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

def compute_metrics(pred_ids, label_ids, processor):
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    # WER
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

def process_data(file_path, image_dir, fraction=1.0, split=True, test_size=0.2):
    """
    Reads, processes, and optionally splits the dataset.
    """
    # Read data
    df = pd.read_csv(file_path, encoding="utf-8", sep='\t', header=None, names=["file_name", "text"])
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
    """
    Creates a dataset instance using the provided dataframe.
    """
    return dataset_class(root_dir=root_dir, processor=processor, df=dataframe)