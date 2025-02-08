train_file_path="/home/nazmuddoha_ansary/work/interspeech2025/archive/train.xlsx"
valid_file_path="/home/nazmuddoha_ansary/work/interspeech2025/archive/valid.xlsx"
vocab_file_path="/home/nazmuddoha_ansary/work/interspeech2025/vocab.json"
train_audio_path="/home/nazmuddoha_ansary/work/interspeech2025/archive/train/"
valid_audio_path="/home/nazmuddoha_ansary/work/interspeech2025/archive/valid"
model_save_dir="/home/nazmuddoha_ansary/work/interspeech2025/"
metrics_file_path="/home/nazmuddoha_ansary/work/interspeech2025/history.csv"
pretrained_model_path="/home/nazmuddoha_ansary/work/interspeech2025/regspeech.pt"

Config = {
    'model_name': 'facebook/wav2vec2-xls-r-300m',
    'lr': 1e-5,
    'wd': 1e-5,
    'T_0': 10,
    'T_mult': 2,
    'eta_min': 1e-6,
    'nb_epochs': 100,
    'train_bs': 24,
    'valid_bs': 24,
    'sampling_rate': 16000,
}

import os
import csv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
) 


import librosa

import warnings
warnings.simplefilter('ignore')
tqdm.pandas()

from jiwer import wer, cer

def calculate_wer_cer(predictions, references):
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER)
    """
    wer_score = wer(references, predictions)
    cer_score = cer(references, predictions)
    return wer_score, cer_score


train_df = pd.read_excel(train_file_path, engine="openpyxl")
train_df["file_name"]=train_df["file_name"].progress_apply(lambda x: os.path.join(train_audio_path,x))
valid_df = pd.read_excel(valid_file_path, engine="openpyxl")
valid_df["file_name"]=valid_df["file_name"].progress_apply(lambda x: os.path.join(valid_audio_path,x))
all_df=pd.concat([train_df,valid_df],ignore_index=True)

def construct_vocab(texts):
    """
    Get unique characters from all the text in a list.
    """
    all_text = " ".join(texts)
    vocab = sorted(list(set(all_text)))
    return vocab

def save_vocab(dataframe, vocab_file_path, column_name='transcripts'):
    """
    Saves the processed vocab file as 'vocab.json', to be ingested by a tokenizer.
    """
    vocab = construct_vocab(dataframe[column_name].tolist())
    vocab_dict = {v: k for k, v in enumerate(vocab)}

    # Ensure the space token is correctly handled
    if " " not in vocab_dict:
        vocab_dict[" "] = len(vocab_dict)

    # Add special tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_file_path, 'w', encoding='utf-8') as fl:
        json.dump(vocab_dict, fl, ensure_ascii=False)

    print("Created Vocab file!")


save_vocab(all_df,vocab_file_path)


# Init the tokenizer, feature_extractor, processor and model
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file_path, 
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token=" "
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, 
    sampling_rate=Config['sampling_rate'], 
    padding_value=0.0, 
    do_normalize=True, 
    return_attention_mask=False
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, 
    tokenizer=tokenizer
)

model = Wav2Vec2ForCTC.from_pretrained(
    Config['model_name'],
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size = len(tokenizer),
    ignore_mismatched_sizes=True
)

if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path))
    print(f"Loaded pretrained model from {pretrained_model_path}")
# Freeze the feature encoder part since we won't be training it
model.to('cuda')
model.freeze_feature_encoder()
for param in model.wav2vec2.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=Config['lr'], 
    weight_decay=Config['wd']
)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=Config['T_0'],
    T_mult=Config['T_mult'],
    eta_min=Config['eta_min']
)


class ASRDataset(Dataset):
    def __init__(self, df, config, is_test=False):
        self.df = df
        self.config = config
        self.is_test = is_test
    
    def __getitem__(self, idx):
        # First read and pre-process the audio file
        audio,_ = librosa.load(self.df.loc[idx]["file_name"], sr=None)
        audio = processor(
            audio, 
            sampling_rate=self.config['sampling_rate']
        ).input_values[0]
        
        # Return -1 for label if in test-only mode
        if self.is_test:
            return {'audio': audio, 'label': -1}
        else:
            # If we are training/validating, also process the labels (actual sentences)
            with processor.as_target_processor():
                labels = processor(self.df.loc[idx]['transcripts']).input_ids
            return {'audio': audio, 'label': labels}
        
    def __len__(self):
        return len(self.df)
    
def ctc_data_collator(batch):
    """
    Custom data collator function to dynamically pad the data
    """
    input_features = [{"input_values": sample["audio"]} for sample in batch]
    label_features = [{"input_ids": sample["label"]} for sample in batch]
    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )
    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch

def train_one_epoch(model, train_loader, optimizer, scheduler, device='cuda:0'):
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))
    avg_loss = 0
    all_predictions = []
    all_references = []
    
    for data in pbar:
        data = {k: v.to(device) for k, v in data.items()}
        outputs = model(**data)
        loss = outputs.loss
        loss_itm = loss.item()
        
        avg_loss += loss_itm
        
        # Decode predictions and references
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        predictions = processor.batch_decode(pred_ids)
        references = processor.batch_decode(data["labels"], group_tokens=False)
        
        all_predictions.extend(predictions)
        all_references.extend(references)
        
        # Calculate WER and CER for the current batch
        batch_wer, batch_cer = calculate_wer_cer(predictions, references)
        
        # Update progress bar description
        pbar.set_description(
            f"loss: {loss_itm:.4f} | WER: {batch_wer:.4f} | CER: {batch_cer:.4f}"
        )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    # Calculate overall WER and CER for the epoch
    epoch_wer, epoch_cer = calculate_wer_cer(all_predictions, all_references)
    print(f"Train WER: {epoch_wer:.4f}, Train CER: {epoch_cer:.4f}")
    
    return avg_loss / len(train_loader), epoch_wer, epoch_cer

@torch.no_grad()
def valid_one_epoch(model, valid_loader, device='cuda:0'):
    model.eval()
    pbar = tqdm(valid_loader, total=len(valid_loader))
    avg_loss = 0
    all_predictions = []
    all_references = []
    
    for data in pbar:
        data = {k: v.to(device) for k, v in data.items()}
        outputs = model(**data)
        loss = outputs.loss
        loss_itm = loss.item()
        
        avg_loss += loss_itm
        
        # Decode predictions and references
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        predictions = processor.batch_decode(pred_ids)
        references = processor.batch_decode(data["labels"], group_tokens=False)
        
        all_predictions.extend(predictions)
        all_references.extend(references)
        
        # Calculate WER and CER for the current batch
        batch_wer, batch_cer = calculate_wer_cer(predictions, references)
        
        # Update progress bar description
        pbar.set_description(
            f"val_loss: {loss_itm:.4f} | WER: {batch_wer:.4f} | CER: {batch_cer:.4f}"
        )
        
    # Calculate overall WER and CER for the epoch
    epoch_wer, epoch_cer = calculate_wer_cer(all_predictions, all_references)
    print(f"Validation WER: {epoch_wer:.4f}, Validation CER: {epoch_cer:.4f}")
    
    return avg_loss / len(valid_loader), epoch_wer, epoch_cer

# Construct training and validation dataloaders
train_ds = ASRDataset(train_df, Config)
valid_ds = ASRDataset(valid_df, Config)

train_loader = DataLoader(
    train_ds, 
    batch_size=Config['train_bs'], 
    collate_fn=ctc_data_collator, 
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=Config['valid_bs'],
    collate_fn=ctc_data_collator,
)

    
    
    
# Train the model
best_loss = float('inf')
for epoch in range(Config['nb_epochs']):
    print(f"{'='*40} Epoch: {epoch+1} / {Config['nb_epochs']} {'='*40}")
    train_loss, train_wer, train_cer = train_one_epoch(model, train_loader, optimizer, scheduler)
    valid_loss, valid_wer, valid_cer = valid_one_epoch(model, valid_loader)
    
    print(f"train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")
    print(f"Train WER: {train_wer:.4f}, Train CER: {train_cer:.4f}")
    print(f"Validation WER: {valid_wer:.4f}, Validation CER: {valid_cer:.4f}")
    
    # Save metrics to CSV
    with open(metrics_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, valid_loss, valid_wer, valid_cer])
    
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), f"{model_save_dir}regspeech.pt")
        print(f"Saved the best model so far with val_loss: {valid_loss:.4f}")