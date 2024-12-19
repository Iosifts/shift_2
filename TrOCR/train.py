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
logging.set_verbosity_error()  # Only errors will be printed
from transformers import GenerationConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import logging
from logging import FileHandler, StreamHandler, Formatter
import util
from dataset import OCRDataset, IAMDataset
import argparse
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser('TrOCR Training')
    parser.add_argument('--data', type=str, help='Search keyword(s) (required)', required=True)
    parser.add_argument('--output', type=str, default='output', help='Search keyword(s) (required)')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--model', default='microsoft/trocr-base-handwritten', 
                        choices=['microsoft/trocr-base-stage1', 'microsoft/trocr-large-stage1', 
                        'microsoft/trocr-base-handwritten'], help='Select the model to use.')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs to train')
    parser.add_argument('--batchsize', type=int, default=4, help='Batchsize of DataLoader')
    parser.add_argument('--val_iters', type=int, default=1, help='Number of epochs to eval at')
    parser.add_argument('--lr', type=float, default=4e-5, help='Learning rate of update step')
    parser.add_argument('--lr_patience', type=int, default=2, help='Patience for lr scheduler')
    parser.add_argument('--num_samples', type=float, default=10, 
                        help='Number of printed sample predictions')
    return parser.parse_args()

if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    datapath = args.data
    base_output_dir = args.output
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = time.strftime("%m%d")
    formatted_lr = util.format_lr(args.lr)
    dataset_name = os.path.basename(os.path.normpath(datapath))
    output_dir_name = f"{dataset_name}_e{args.epochs}_lr{formatted_lr}_b{args.batchsize}_{timestamp}"
    output_dir = os.path.join(base_output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    """Logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = StreamHandler()
    file_handler = FileHandler(os.path.join(output_dir, "training.log"), encoding='utf-8', 
                               errors='replace') # Python will raise an error if it encounters a character it cannot encode
    formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Initializing Logging")

    """Data Processing"""
    custom = True # use custom dataset
    change_eval = False # use different val dataset, instead of splitting
    evalpath = 'data/balcesu_test'
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") # for data processing
    fraction = 1.0 # fraction of train dataset used
    eval_fraction = 0.25

    if custom:
        train_file = os.path.join(datapath, "labels.txt")
        train_df, test_df = util.process_data(train_file, os.path.join(datapath, "images"), fraction=fraction)
        train_dataset = util.create_dataset(train_df, datapath, processor, OCRDataset)

        if change_eval:
            eval_file = os.path.join(evalpath, "labels.txt")
            _, eval_df = util.process_data(eval_file, os.path.join(evalpath, "images"), fraction=eval_fraction, split=False)
            eval_dataset = util.create_dataset(eval_df, evalpath, processor, OCRDataset)
        else:
            eval_dataset = util.create_dataset(test_df, datapath, processor, OCRDataset)
    else:
        root_dir = "path/to/IAM/"
        dataset_url = "https://fki.tic.heia-fr.ch/DBs/iamDB/data/words.tgz"
        labels_file = os.path.join(root_dir, "labels.txt")

        train_df, test_df = util.process_data(labels_file, os.path.join(root_dir, "images"), fraction=fraction)
        train_dataset = util.create_dataset(train_df, root_dir, processor, IAMDataset)
        eval_dataset = util.create_dataset(test_df, root_dir, processor, IAMDataset)

    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of validation examples: {len(eval_dataset)}")
    
    # Verify dataset encoding
    encoding = train_dataset[0]
    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.decode(labels, skip_special_tokens=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batchsize)

    """Model Initialization"""
    logger.info(f"Model '{args.model}' selected.")
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.to(device)

    # Add additional tokens
    special_chars = ["ă", "â", "î", "ș", "ț", "Ă", "Â", "Î", "Ș", "Ț"]
    processor.tokenizer.add_tokens(special_chars, special_tokens=False)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Set special tokens for training and ensure model config consistency
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id  # 0 <s>
    model.config.eos_token_id = processor.tokenizer.eos_token_id            # 2 </s>
    model.config.pad_token_id = processor.tokenizer.pad_token_id            # 1 <pad>
    model.config.vocab_size = len(processor.tokenizer)

    generation_config = GenerationConfig(
        eos_token_id=model.config.eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=model.config.pad_token_id,
        max_length=105,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        num_beams=4,
    )
    model.generation_config = generation_config

    """Training Setup"""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience)
    train_losses = []
    val_losses = []
    best_val_loss = 1e6
    best_val_char_acc = -1.0

    # Load from (.pt) checkpoint if specified
    start_epoch = 0
    if args.checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        start_epoch = checkpoint['epoch']  # epoch at which it was saved
        
        logger.info(f"Resumed at epoch {start_epoch} with best_val_loss {best_val_loss:.4f}")

    """Training"""
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}", dynamic_ncols=True):
            for k,v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Train loss after epoch {epoch}: {avg_train_loss}")

        # Evaluation
        if epoch % args.val_iters == 0:
            model.eval()
            total_cer = 0.0
            total_wer = 0.0
            total_acc = 0.0
            total_char_acc = 0.0
            total_lev_dist = 0.0
            total_val_loss = 0.0
            count = 0

            sample_preds = []
            sample_refs = []
            sample_confs = []

            with torch.no_grad():
                for batch in tqdm(eval_dataloader, dynamic_ncols=True):
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)

                    # Compute validation loss
                    val_outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss = val_outputs.loss.item()
                    total_val_loss += val_loss

                    # Generate predictions with scores
                    generation_outputs = model.generate(
                        pixel_values, 
                        output_scores=True, 
                        return_dict_in_generate=True
                    )

                    pred_ids = generation_outputs.sequences
                    scores_list = generation_outputs.scores  # List of logits for each generated token step

                    cer, wer, acc, char_acc, lev_dist = util.compute_metrics(
                        pred_ids=pred_ids, label_ids=labels, processor=processor
                    )

                    total_cer += cer
                    total_wer += wer
                    total_acc += acc
                    total_char_acc += char_acc
                    total_lev_dist += lev_dist
                    count += 1

                    # Collect samples until we reach num_samples
                    if len(sample_preds) < args.num_samples:
                        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

                        labels_adj = labels.clone()
                        labels_adj[labels_adj == -100] = processor.tokenizer.pad_token_id
                        label_str = processor.batch_decode(labels_adj, skip_special_tokens=True)

                        batch_size = pred_ids.size(0)
                        seq_len = pred_ids.size(1)
                        sample_needed = args.num_samples - len(sample_preds)
                        sample_size = min(batch_size, sample_needed)

                        # Compute confidence scores for these samples
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

                        # Stop collecting
                        if len(sample_preds) >= args.num_samples:
                            # No need to process further batches for sample printing
                            # But we still complete the evaluation loop to compute metrics over all batches if desired
                            pass

            avg_val_loss = total_val_loss / count
            avg_val_cer = total_cer / count
            avg_val_wer = total_wer / count
            avg_val_acc = total_acc / count
            avg_val_char_acc = total_char_acc / count
            avg_val_lev_dist = total_lev_dist / count
            val_losses.append(avg_val_loss)

            # LR scheduler steps on val loss
            scheduler.step(avg_val_loss)

            # Log metrics
            logger.info(
                f"Epoch {epoch} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"CER: {avg_val_cer:.4f} | "
                f"WER: {avg_val_wer:.4f} | "
                f"Acc: {avg_val_acc:.4f} | "
                f"CharAcc: {avg_val_char_acc:.4f} | "
                f"LevDist: {avg_val_lev_dist:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e}"
            )

            # Log sample predictions
            if sample_preds and sample_refs:
                logger.info("Sample Predictions:")
                logger.info("-" * 80)
                logger.info(f"{'Prediction':<35} | {'Reference':<35} | {'Confidence':<8}")
                logger.info("-" * 80)
                for p, r, c in zip(sample_preds, sample_refs, sample_confs):
                    logger.info(f"{p[:35]:<35} | {r[:35]:<35} | {c:.4f}")
                logger.info("-" * 80)

            # Save checkpoint if it's the best val_loss
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
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, best_checkpoint_path)
                logger.info(f"New best checkpoint saved to {best_checkpoint_path} with Val Loss: {best_val_loss:.4f}")

            # Save checkpoint if it's the best char_acc
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
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, best_checkpoint_char_acc_path)
                logger.info(f"New best checkpoint saved to {best_checkpoint_char_acc_path} with CharAcc: {best_val_char_acc:.4f}")

        else:
            # Not validating this epoch
            pass

    """Plotting & Saving"""
    val_epochs = [i * args.val_iters for i in range(len(val_losses))]
    val_epochs = [ve for ve in val_epochs if ve < len(train_losses)]
    max_len = min(len(val_epochs), len(val_losses))
    val_epochs = val_epochs[:max_len]
    val_losses = val_losses[:max_len]
    if len(val_epochs) != len(val_losses):
        logger.warning("Still mismatch in val_losses and val_epochs lengths after trimming!")
    plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, [train_losses[i] for i in val_epochs], label='Train Loss (sampled)')
    plt.plot(val_epochs, val_losses, label='Val Loss')
    plt.title('Training and Validation Loss over Selected Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_validation_loss.pdf"))
    plt.show()

    # Save in Hugging Face format
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
