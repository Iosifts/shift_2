import torch
from torch import nn
from transformers import AutoModel
from PIL import Image

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
