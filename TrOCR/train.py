import os
import argparse
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from tqdm import tqdm
from transformers import logging
logging.set_verbosity_error()
from collections import defaultdict
missing_tokens_freq = defaultdict(int)

from util import (
    plot_metric, set_seed, load_config, 
    setup_directories, save_checkpoint,
    create_logger, update_tokenizer,
    load_checkpoint, compute_metrics,
    custom_decode,
)
from models import (
    get_processor_and_model, train_step,
    generate_step, calculate_confidence_scores,
)
from dataset import (
    init_dataset,
)

def parse_args():
    """Implements CLI and config file system"""
    p = argparse.ArgumentParser('TrOCR Training')
    
    # Config override
    p.add_argument('--config', type=str, default=None, help='YAML config file to override arguments')

    # Data params
    p.add_argument('--data', type=str, help='Directory with labeled image data')
    p.add_argument('--dataset', default='custom', choices=['IAM', 'MNIST', 'custom'])
    p.add_argument('--fraction', type=float, default=1.0, help='Dataset fraction for splits')
    p.add_argument('--testdata', type=str, default=None, help='Optional separate test data directory')
    p.add_argument('--test_frac', type=float, default=1.0, help='Test data fraction if --testdata used')
    
    # Model params
    p.add_argument('--model', default='microsoft/trocr-large-handwritten', choices=[
        'microsoft/trocr-base-stage1', 'microsoft/trocr-large-stage1',
        'microsoft/trocr-base-handwritten', 'microsoft/trocr-large-handwritten',
        'microsoft/trocr-small-handwritten', 'microsoft/trocr-base-printed',
        'microsoft/trocr-small-printed', 'naver-clova-ix/donut-base',
        'naver-clova-ix/donut-base-finetuned-rvlcdip', 'naver-clova-ix/donut-proto',
        'facebook/nougat-base', 'facebook/nougat-small',
        'microsoft/dit-base', 'microsoft/dit-large', 'custom'
    ])
    p.add_argument('--tokenizer', type=str, default=None, help='Overwrite processor tokenizer')
    p.add_argument('--special_chars', nargs='+', default=None, help='Add special characters to tokenizer')
    p.add_argument('--checkpoint', type=str, default=None, help='Load checkpoint')

    # Training params
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batchsize', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    p.add_argument('--lr_patience', type=int, default=2, help='LR scheduler patience')
    p.add_argument('--lr_factor', type=int, default=0.5, help='LR scheduler reduction')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_iters', type=int, default=1, help='Eval every N epochs')
    p.add_argument('--test_iters', type=int, default=2, help='Test every N epochs')
    p.add_argument('--num_samples', type=float, default=10, help='Samples to print')

    # Output params
    p.add_argument('--output', type=str, default='data/output', help='Output directory')
    p.add_argument('--timestamp', type=bool, default=False, help='Add timestamp to output dir')
    p.add_argument('--save_predictions', type=bool, default=False, help='Save predictions to file')
    p.add_argument('--suffix', type=str, default='', help='Suffix for the output directory')

    # Add augmentation flag
    p.add_argument('--use_augmentation', type=bool, default=False,
                  help='Whether to use data augmentation during training')

    args = p.parse_args()
    if args.config:
        config = load_config(args.config)
        args_dict = vars(args)
        args_dict.update({k: v for k, v in config.items() if v is not None})
    return args

def train(args):
    """Main training function"""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup directories and logging
    model_name, output_dir, predictions_dir = setup_directories(args)
    logger = create_logger(output_dir)

    # Initialize model and data processing
    processor, model = get_processor_and_model(args, logger)
    model.to(device)

    # Create datasets and dataloaders
    datasets = init_dataset(args, processor, args.fraction, args.test_frac, logger)
    train_dataloader = DataLoader(datasets['train'], batch_size=args.batchsize, shuffle=True)
    eval_dataloader = DataLoader(datasets['eval'], batch_size=args.batchsize)
    test_dataloader = DataLoader(datasets['test'], batch_size=args.batchsize)

    # Update tokenizer with special characters
    char_to_token_map, token_to_char_map = update_tokenizer(
        processor, model, args.special_chars, logger)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)

    # Initialize tracking variables
    metrics = {
        'train': {'loss': []},
        'val': {'loss': [], 'char_acc': [], 'wer': []},
        'test': {'loss': [], 'char_acc': [], 'wer': []}
    }
    best_metrics = {'loss': float('inf'), 'char_acc': -1.0}
    global_step = 0
    epoch = 1
    
    # Load checkpoint if specified
    if args.checkpoint:
        epoch, best_metrics, metrics['train'], metrics['val'], metrics['test'] = \
            load_checkpoint(args.checkpoint, model, optimizer, scheduler, device, logger)
        args.epochs = epoch + args.epochs

    # Train loop
    while epoch <= args.epochs:
        # Training phase
        model.train()
        running_train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}", dynamic_ncols=True):
            loss = train_step(model, batch, device, args.model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_train_loss += loss.item()
            metrics['train']['loss'].append((global_step, loss.item()))
            global_step += 1

        avg_train_loss = running_train_loss / len(train_dataloader)
        logger.info(f"Train loss after epoch {epoch}: {avg_train_loss}")

        # Validation phase  
        if epoch % args.val_iters == 0:
            val_metrics = evaluate(model, optimizer, eval_dataloader, processor, device, args, 
                token_to_char_map, logger, epoch, prefix="EVAL")
            metrics['val']['loss'].append((global_step, val_metrics['loss']))
            metrics['val']['wer'].append((global_step, val_metrics['wer']))
            metrics['val']['char_acc'].append((global_step, val_metrics['char_acc']))

            scheduler.step(val_metrics['loss'])
        
        # Test phase
        if epoch % args.test_iters == 0:
            test_metrics = evaluate(model, optimizer, test_dataloader, processor, device, args, 
                token_to_char_map, logger, epoch, predictions_dir, prefix="TEST")
            metrics['test']['loss'].append((global_step, test_metrics['loss']))
            metrics['test']['wer'].append((global_step, test_metrics['wer']))
            metrics['test']['char_acc'].append((global_step, test_metrics['char_acc']))

            # Plot metrics
            plot_metric(metrics['train']['loss'], metrics['val']['loss'], metrics['test']['loss'],
                metric_name="loss", output_dir=output_dir, logger=logger, use_log_scale=True)
            plot_metric(
                [], [], metrics['test']['wer'], metric_name="wer", output_dir=output_dir, 
                logger=logger, show_epoch=True, iters_per_epoch=len(train_dataloader))
            plot_metric(
                [], [], metrics['test']['char_acc'], metric_name="char_acc", output_dir=output_dir, 
                logger=logger, show_epoch=True, iters_per_epoch=len(train_dataloader))

            # Save checkpoint
            best_metrics = save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, metrics=test_metrics, best_metrics=best_metrics,
                train_metrics=metrics['train'], val_metrics=metrics['val'],
                test_metrics=metrics['test'], output_dir=output_dir,
                model_name=model_name, logger=logger
            )

        epoch += 1

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def evaluate(model, optimizer,dataloader, processor, device, args, 
             token_to_char_map, logger, epoch, predictions_dir=None, prefix="EVAL"):
    """Run evaluation loop"""
    model.eval()
    metrics_totals = {
        'loss': 0.0, 'cer': 0.0, 'wer': 0.0, 
        'acc': 0.0, 'char_acc': 0.0, 'lev_dist': 0.0
    }
    count = 0
    sample_preds, sample_refs, sample_confs = [], [], []

    if prefix == "TEST":
        all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{prefix}", dynamic_ncols=True):
            # Forward pass
            val_loss = train_step(model, batch, device, args.model)
            generation_outputs = generate_step(model, batch, device, args.model, model.generation_config)
            pred_ids = generation_outputs.sequences
            
            # Prepare labels
            labels_adj = batch["labels"].clone()
            labels_adj[labels_adj == -100] = processor.tokenizer.pad_token_id
            
            # Decode predictions and labels
            label_str = processor.batch_decode(labels_adj, skip_special_tokens=True)
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

            # Custom decode predictions and labels
            # pred_str = []
            # for ids in pred_ids:
            #     decoded = custom_decode(
            #         ids.tolist(), processor.tokenizer,
            #         token_to_char_map, logger)
            #     pred_str.append(decoded)
            
            # label_str = []
            # for ids in labels_adj:
            #     decoded = custom_decode(
            #         ids.tolist(), processor.tokenizer,
            #         token_to_char_map, logger)
            #     label_str.append(decoded)

            if len(pred_str) != len(label_str):
                pred_str = pred_str * len(label_str) if len(pred_str) == 1 else pred_str[:len(label_str)]
            
            # Save test predictions
            if prefix == "TEST" and args.save_predictions:
                scores_list = generation_outputs.scores
                confidences = calculate_confidence_scores(
                    pred_ids=pred_ids, scores_list=scores_list, 
                    sample_size=len(pred_str)
                )
                for pred, ref, conf, ids in zip(pred_str, label_str, confidences, pred_ids):
                    all_predictions.append({
                        'prediction': pred,
                        'reference': ref,
                        'confidence': float(conf),
                        'token_ids': ids.tolist()
                    })

            # Compute metrics
            metrics_totals['loss'] += val_loss
            batch_metrics = compute_metrics(pred_str, label_str, logger)
            for name, value in zip(['cer', 'wer', 'acc', 'char_acc', 'lev_dist'], batch_metrics):
                metrics_totals[name] += value
            count += 1

            # Collect samples
            scores_list = generation_outputs.scores
            if len(sample_preds) < args.num_samples:
                batch_size = pred_ids.size(0)
                sample_needed = args.num_samples - len(sample_preds)
                sample_size = min(batch_size, sample_needed)
                sample_confs.extend(calculate_confidence_scores(
                    pred_ids=pred_ids, scores_list=scores_list, sample_size=sample_size))
                sample_preds.extend(pred_str[:sample_size])
                sample_refs.extend(label_str[:sample_size])

    # Calculate averages
    metrics_avg = {
        k: float(v / count) if count > 0 else 0.0
        for k, v in metrics_totals.items()
    }

    # Log results
    logger.info(
        f"[{prefix}] | "
        f"Loss: {metrics_avg['loss']:.4f} | "
        f"CER: {metrics_avg['cer']:.4f} | "
        f"WER: {metrics_avg['wer']:.4f} | "
        f"Acc: {metrics_avg['acc']:.4f} | "
        f"CharAcc: {metrics_avg['char_acc']:.4f} | "
        f"LevDist: {metrics_avg['lev_dist']:.4f}"
        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Save test predictions to file
    if prefix == "TEST" and args.save_predictions:
        predictions_file = os.path.join(predictions_dir, f'test_preds_e{epoch}.json')
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump({
                'predictions': all_predictions,
                'metrics': metrics_avg,
                'model': args.model,
                'epoch': epoch,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(all_predictions)} test predictions to {predictions_file}")

    # Log samples
    if sample_preds and sample_refs:
        logger.info(f"[{prefix}] Sample Predictions:")
        logger.info("-" * 80)
        logger.info(f"{'Prediction':<35} | {'Reference':<35} | {'Confidence':<8}")
        logger.info("-" * 80)
        for p, r, c in zip(sample_preds, sample_refs, sample_confs):
            logger.info(f"{p[:35]:<35} | {r[:35]:<35} | {c:.4f}")
        logger.info("-" * 80)

    return metrics_avg

if __name__ == "__main__":
    args = parse_args()
    train(args)
