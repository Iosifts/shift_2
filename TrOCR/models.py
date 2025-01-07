import torch
from torch import nn
from PIL import Image
from transformers import (
    AutoModel, DonutProcessor, AutoImageProcessor, AutoTokenizer,
    VisionEncoderDecoderModel, TrOCRProcessor, DonutProcessor,
    NougatProcessor, GenerationConfig,
)
from typing import Dict
from torch import Tensor

# ---------------------------
### Fine-tuned model handling
# ---------------------------

def get_config(args):
    """Configure model-specific parameters"""
    
    config = {
        'generation_config': None,
        'optimizer_params': {'lr': args.lr},
        'scheduler_params': {'patience': args.lr_patience},
        'training_params': {}
    }
    
    if 'microsoft/trocr' in args.model:
        config['generation_config'] = GenerationConfig(
            max_length=105,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5, # 2
            num_beams=4, # 4
            bos_token_id=0,
            decoder_start_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
        )
        
    elif 'naver-clova-ix/donut' in args.model:
        config['generation_config'] = GenerationConfig(
            max_length=105,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            bos_token_id=0,
            decoder_start_token_id=0,
            eos_token_id=2,
            pad_token_id=1
        )
        config['optimizer_params']['lr'] = 5e-5  # Donut preferred learning rate
        
    elif 'facebook/nougat' in args.model:
        config['generation_config'] = GenerationConfig(
            max_length=105,
            num_beams=4,
            length_penalty=1.0,
            do_sample=True,
            temperature=0.7,
            decoder_start_token_id=0,
            bos_token_id=0,
            pad_token_id=1,
            eos_token_id=2
        )
        config['training_params']['gradient_checkpointing'] = True
        
    elif 'microsoft/dit' in args.model:
        config['generation_config'] = GenerationConfig(
            max_length=196,
            num_beams=5
        )
        config['optimizer_params']['weight_decay'] = 0.05

    elif args.model == 'custom':
        config['generation_config'] = GenerationConfig(
            max_length=105,
            num_beams=4,
            length_penalty=1.0,
            do_sample=True,
            temperature=0.7,
            decoder_start_token_id=0,
            bos_token_id=0,
            pad_token_id=1,
            eos_token_id=2
        )
        # Custom model specific parameters
        config['optimizer_params'].update({
            'lr': 5e-5,  # Typically need higher LR for custom models
            'weight_decay': 0.01
        })
        config['scheduler_params'].update({
            'patience': 3,  # More patience for custom architecture
            'factor': 0.5
        })
        config['training_params'].update({
            'gradient_checkpointing': True,  # Memory efficiency
            'warmup_steps': 100
        })
       
    return config

def get_processor_and_model(args, logger):
    """Initialize appropriate processor and model based on selection"""

    if not hasattr(args, 'model'):
        raise ValueError("args must contain 'model' attribute")
    
    if args.model not in [
        'microsoft/trocr-base-stage1', 
        'microsoft/trocr-large-stage1',
        'microsoft/trocr-base-handwritten', 
        'microsoft/trocr-large-handwritten',
        'microsoft/trocr-small-handwritten',
        'microsoft/trocr-base-printed',
        'microsoft/trocr-small-printed',
        'naver-clova-ix/donut-base',
        'naver-clova-ix/donut-base-finetuned-rvlcdip',
        'naver-clova-ix/donut-proto',
        'facebook/nougat-base',
        'facebook/nougat-small',
        'microsoft/dit-base',
        'microsoft/dit-large',
        'custom'
    ]:
        raise ValueError(f"Unsupported model: {args.model}")
    
    config = get_config(args)
    
    if 'microsoft/trocr' in args.model:
        processor = TrOCRProcessor.from_pretrained(args.model)
        original_tokenizer = processor.tokenizer

        if args.tokenizer:
            # Overwrite tokenizer for the language:
            # i.e. dumitrescustefan/bert-base-romanian-cased-v1 for romanian language
            new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            new_tokenizer.add_special_tokens({
                'bos_token': '[CLS]',
                'eos_token': '[SEP]',
                'pad_token': '[PAD]',
                'unk_token': '[UNK]'
            })
            processor.tokenizer = new_tokenizer

        model = VisionEncoderDecoderModel.from_pretrained(args.model)
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
        model.config.bos_token_id = processor.tokenizer.bos_token_id
        model.config.eos_token_id = processor.tokenizer.eos_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = len(processor.tokenizer)
        
    elif 'naver-clova-ix/donut' in args.model:
        processor = DonutProcessor.from_pretrained(args.model)
        model = VisionEncoderDecoderModel.from_pretrained(args.model)
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
        model.config.bos_token_id = processor.tokenizer.bos_token_id
        model.config.eos_token_id = processor.tokenizer.eos_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
    elif 'facebook/nougat' in args.model:
        processor = NougatProcessor.from_pretrained(args.model)
        model = VisionEncoderDecoderModel.from_pretrained(args.model)
        processor.max_source_positions = 4096
        processor.max_target_positions = 1024
        model.config.decoder_start_token_id = 0
        model.config.bos_token_id = 0
        model.config.forced_bos_token_id = 0
        model.config.eos_token_id = 2
        model.config.pad_token_id = 1
        
    elif args.model == 'custom':
        image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
        processor = CustomProcessor(image_processor, tokenizer)

        vision_encoder = AutoModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        text_encoder = CustomEncoder(pretrained_bert_model='dumitrescustefan/bert-base-romanian-cased-v1')
        vocab_size = len(processor.tokenizer)
        hidden_size = text_encoder.encoder.config.hidden_size
        text_decoder = CustomDecoder(vocab_size, hidden_size)
        model = CustomVisionEncoderDecoderModel(vision_encoder, text_encoder, text_decoder, vocab_size)

        model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        
    if hasattr(model, 'config'):
        required_tokens = ['decoder_start_token_id', 'bos_token_id', 'eos_token_id', 'pad_token_id']
        token_values = {
            token: getattr(model.config, token) 
            for token in required_tokens
        }
        
        # Verify all required tokens are set
        missing_tokens = [token for token, value in token_values.items() if value is None]
        if missing_tokens:
            raise ValueError(f"Missing required token IDs: {missing_tokens}")
            
        # logger.info(f"Token IDs set: bos={token_values['bos_token_id']}, "
        #            f"decoder_start={token_values['decoder_start_token_id']}, "
        #            f"eos={token_values['eos_token_id']}, "
        #            f"pad={token_values['pad_token_id']}, "
        #            f"vocab_size={model.config.vocab_size}")

    if config['generation_config'] is not None:
        model.generation_config = config['generation_config']
        logger.info(f"Using {model.generation_config}")

    return processor, model

def train_step(model: nn.Module, 
               batch: Dict[str, Tensor], 
               device: torch.device, 
               model_type: str) -> Tensor:
    """Handle different model output formats during training"""
    batch = {k: v.to(device) for k, v in batch.items()}
    
    if model_type == 'custom':
        outputs = model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]  # Only pass labels
        )
        loss = outputs['loss']
    elif 'microsoft/trocr' in model_type:
        outputs = model(**batch)
        loss = outputs.loss
    elif 'naver-clova-ix/donut' in model_type:
        outputs = model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        loss = outputs.loss
    elif 'facebook/nougat' in model_type:
        outputs = model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        loss = outputs.loss
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return loss

def generate_step(model, batch, device, model_type, generation_config):
    """Handle different model generation approaches"""
    pixel_values = batch["pixel_values"].to(device)
    
    if 'microsoft/trocr' in model_type:
        outputs = model.generate(
            pixel_values,
            generation_config=generation_config,
            output_scores=True,
            return_dict_in_generate=True
        )
    elif 'naver-clova-ix/donut' in model_type:
        outputs = model.generate(
            pixel_values=pixel_values,
            generation_config=generation_config,
            return_dict_in_generate=True
        )
        outputs.scores = None # not supported by donut
    elif 'facebook/nougat' in model_type:
        outputs = model.generate(
            pixel_values,
            generation_config=generation_config,
            output_scores=True,
            return_dict_in_generate=True
        )
    else:
        raise ValueError(f"Unsupported model type for generation: {model_type}")
    
    # Ensure we have a batch dimension
    if not hasattr(outputs, 'sequences'):
        outputs = type('GenerationOutput', (), {
            'sequences': outputs.unsqueeze(0) if outputs.dim() == 1 else outputs,
            'scores': None
        })
    elif outputs.sequences.dim() == 1:
        outputs.sequences = outputs.sequences.unsqueeze(0)

    return outputs

def calculate_confidence_scores(pred_ids, scores_list, sample_size):
    """
    Calculate confidence scores for generated sequences
    
    Args:
        pred_ids: Tensor of predicted token IDs
        scores_list: List of score tensors from model generation
        sample_size: Number of samples to process
    
    Returns:
        list: Confidence scores for each sample
    """
    sample_confs = []
    
    if scores_list is not None:
        for i in range(sample_size):
            token_ids = pred_ids[i]
            token_probs = []
            seq_len = token_ids.size(0)
            
            for step_idx in range(seq_len - 1):
                logits = scores_list[step_idx][i]
                probs = torch.softmax(logits, dim=-1)
                chosen_token_id = token_ids[step_idx+1]
                token_prob = probs[chosen_token_id].item()
                token_probs.append(token_prob)
                
            confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
            sample_confs.append(confidence)
    else:
        # For models that don't support confidence scores (like Donut)
        sample_confs.extend([float('nan')] * sample_size)
        
    return sample_confs

# ---------------------------
### Custom Model
# ---------------------------

class CustomProcessor:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images, text=None, return_tensors="pt", padding="max_length", 
                 max_length=None, truncation=True):
        """Make the processor callable like HuggingFace processors"""
        if text is None:
            # Image-only processing
            return self.image_processor(
                images=images, 
                return_tensors=return_tensors
            )
        
        # Process both image and text
        pixel_values = self.image_processor(
            images=images, 
            return_tensors=return_tensors
        )["pixel_values"]
        
        labels = self.tokenizer(
            text_target=text,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors
        ).input_ids

        # Match HuggingFace processor output format
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        """Decode a batch of token IDs to text"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def preprocess_image(image_path, image_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

def preprocess_text(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return inputs["input_ids"], inputs["attention_mask"]

def decode_predictions(pred_ids, tokenizer):
    return tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

class CustomEncoder(nn.Module):
    def __init__(self, pretrained_bert_model: str = 'dumitrescustefan/bert-base-romanian-cased-v1'):
        super(CustomEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_bert_model)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return encoder_outputs.last_hidden_state

class CustomDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=6, num_heads=8, max_len=512):
        super(CustomDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = nn.Embedding(max_len, hidden_size)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_embedded = self.embedding(tgt) + self.positional_encoding(torch.arange(tgt.size(1), device=tgt.device))
        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        logits = self.linear(output)
        return logits

class CustomVisionEncoderDecoderModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, text_decoder, vocab_size):
        super(CustomVisionEncoderDecoderModel, self).__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.vocab_size = vocab_size
        
        # Add decoder property to match HuggingFace interface
        self.decoder = self.text_decoder
        
    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        batch_size = pixel_values.size(0)
        
        # Encode vision features
        vision_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        
        # Create dummy text features if input_ids not provided
        if input_ids is None and labels is not None:
            input_ids = labels
            
        if input_ids is not None:
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
                
            # Encode text
            text_features = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
        else:
            # During inference, we don't have text input
            device = pixel_values.device
            text_features = torch.zeros(
                (batch_size, 1, self.text_encoder.encoder.config.hidden_size), 
                device=device
            )

        # Decode
        logits = self.text_decoder(
            tgt=input_ids if input_ids is not None else labels,
            memory=vision_features
        )

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape logits and labels for loss computation
            loss = loss_fn(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        return {'loss': loss, 'logits': logits} if loss is not None else logits
    
    @property
    def decoder(self):
        return self.text_decoder
    
    @decoder.setter
    def decoder(self, value):
        self.text_decoder = value
    