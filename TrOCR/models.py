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

def get_processor_and_model(args):
    """Initialize appropriate processor and model based on selection"""

    if not hasattr(args, 'model'):
        raise ValueError("args must contain 'model' attribute")
    
    if args.model not in [
        # TrOCR models (working)
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
        # Custom model
        'custom'
    ]:
        raise ValueError(f"Unsupported model: {args.model}")
    
    if 'microsoft/trocr' in args.model:
        processor = TrOCRProcessor.from_pretrained(args.model)
        model = VisionEncoderDecoderModel.from_pretrained(args.model)
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
        model.config.bos_token_id = processor.tokenizer.bos_token_id
        model.config.eos_token_id = processor.tokenizer.eos_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = len(processor.tokenizer)
        
    elif 'naver-clova-ix/donut' in args.model:
        processor = DonutProcessor.from_pretrained(args.model)
        model = VisionEncoderDecoderModel.from_pretrained(args.model)
        processor.task_start_token = "<s>"
        processor.task_end_token = "</s>"
        
    elif 'facebook/nougat' in args.model:
        processor = NougatProcessor.from_pretrained(args.model)
        model = VisionEncoderDecoderModel.from_pretrained(args.model)
        processor.max_source_positions = 4096
        processor.max_target_positions = 1024
        # Explicitly set all token IDs
        model.config.decoder_start_token_id = 0
        model.config.bos_token_id = 0
        model.config.forced_bos_token_id = 0
        model.config.eos_token_id = 2
        model.config.pad_token_id = 1
        
    elif args.model == 'custom':
        # Your existing custom processor code
        # image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        # tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
        # processor = CustomProcessor(image_processor, tokenizer)
        # vision_encoder = AutoModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        # text_encoder = CustomEncoder(pretrained_bert_model='dumitrescustefan/bert-base-romanian-cased-v1')
        # vocab_size = len(processor.tokenizer)
        # hidden_size = text_encoder.encoder.config.hidden_size
        # text_decoder = CustomDecoder(vocab_size, hidden_size)
        # model = CustomVisionEncoderDecoderModel(vision_encoder, text_encoder, text_decoder, vocab_size)
        raise NotImplementedError('Todo: Test pretraining')
        
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

    return processor, model

def train_step(model: nn.Module, 
               batch: Dict[str, Tensor], 
               device: torch.device, 
               model_type: str) -> Tensor:
    """
    Handle different model output formats during training
    
    Args:
        model: The model to train
        batch: Dictionary containing input tensors
        device: Device to run on
        model_type: Type/name of the model being used
        
    Returns:
        Tensor containing the loss value
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    if 'microsoft/trocr' in model_type:
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
            task_prompt="<s>",
            generation_config=generation_config
        )
    elif 'facebook/nougat' in model_type:
        sequences = model.generate(
            pixel_values,
            generation_config=generation_config,
            output_scores=True,
            return_dict_in_generate=True
        )
        # Convert to same format as other models
        outputs = type('GenerationOutput', (), {
            'sequences': sequences.sequences if hasattr(sequences, 'sequences') else sequences,
            'scores': sequences.scores if hasattr(sequences, 'scores') else None
        })
    elif 'custom' in model_type:
        outputs = model.generate(
            pixel_values,
            generation_config=generation_config
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

def get_model_specific_config(args):
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
            max_length=256,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True
        )
        config['optimizer_params']['lr'] = 5e-5  # Donut preferred learning rate
        
    elif 'alibaba-damo/OCR-MASTER' in args.model:
        config['generation_config'] = GenerationConfig(
            max_length=150,
            num_beams=3,
            length_penalty=1.2
        )
        config['training_params']['warmup_steps'] = 1000
        
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
        
    return config

# ---------------------------
### Custom Model Pretraining
# ---------------------------

class CustomProcessor:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def process_image(self, image):
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"]

    def process_text(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

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

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # Encode vision features
        vision_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state

        # Encode text
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Decode using the custom decoder
        logits = self.text_decoder(labels, memory=vision_features)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        return {'loss': loss, 'logits': logits}
    