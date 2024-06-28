import torch.nn as nn
import torch
from transformers import T5Tokenizer, T5EncoderModel
import torch.nn.functional as F

class KGEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = T5EncoderModel.from_pretrained('t5-small')
        self.encoder2 = T5EncoderModel.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.max_length = 1024

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def prepare_source_vector(self, chunk_text:str, device):
        tokenized_chunk = self.tokenizer(chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(device)
        chunk_ids = tokenized_chunk['input_ids'].to(device)
        chunk_attention_mask = tokenized_chunk['attention_mask'].to(device)

        return self.forward(chunk_ids, chunk_attention_mask, encoder=1, inference=True)
    
    def prepare_target_vector(self, chunk_text:str, device):
        tokenized_chunk = self.tokenizer(chunk_text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(device)
        chunk_ids = tokenized_chunk['input_ids'].to(device)
        chunk_attention_mask = tokenized_chunk['attention_mask'].to(device)

        return self.forward(chunk_ids, chunk_attention_mask, encoder=2, inference=True)
        
    def forward(self, chunk_ids, chunk_attention_mask, encoder: int, inference: bool):
        if inference:
            with torch.no_grad():
                if encoder == 1:
                    encoder_outputs = self.encoder1(input_ids=chunk_ids, attention_mask=chunk_attention_mask)
                elif encoder == 2:
                    encoder_outputs = self.encoder2(input_ids=chunk_ids, attention_mask=chunk_attention_mask)
                else:
                    raise

                pooled_output = self.mean_pooling(encoder_outputs, chunk_attention_mask)
                normalized_output = F.normalize(pooled_output, p=2, dim=1)
                return normalized_output.cpu().numpy()  # Convert tensor to numpy array
        else:
            if encoder == 1:
                encoder_outputs = self.encoder1(input_ids=chunk_ids, attention_mask=chunk_attention_mask)
            elif encoder == 2:
                encoder_outputs = self.encoder2(input_ids=chunk_ids, attention_mask=chunk_attention_mask)
            else:
                raise

            pooled_output = self.mean_pooling(encoder_outputs, chunk_attention_mask)
            normalized_output = F.normalize(pooled_output, p=2, dim=1)
            return normalized_output