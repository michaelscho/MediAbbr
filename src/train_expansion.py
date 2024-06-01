from transformers import T5ForConditionalGeneration, T5Config
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from datasets import load_dataset
from torch.utils.data.dataset import random_split

import torch
import csv
import os

class CustomDataset(Dataset):

    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        with open(os.path.join(os.getcwd(),'..','data','training','seq2seq','seq2seq_gt.txt')) as file:
            reader = csv.reader(file, delimiter='\t') # Assuming data is tab separated
            for row in reader:
                # Assuming the first column is abbreviated text and the second is expanded text
                if len(row) == 2:
                    self.data.append((row[0], row[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        abbreviated, expanded = self.data[idx]

        # Encoding the abbreviated text
        source_encoded = self.tokenizer.encode(abbreviated)
        #print(source_encoded.ids)  # Check the encoded token IDs
        #print(source_encoded.tokens)  # Check the tokens

        source_tokenized = source_encoded.ids[:self.max_len]  # Truncate/pad as necessary
        source_attention_mask = [1] * len(source_tokenized)

        # Encoding the expanded text
        target_encoded = self.tokenizer.encode(expanded)
        target_tokenized = target_encoded.ids[:self.max_len]  # Truncate/pad as necessary

        # Padding to max_len
        padding_length = self.max_len - len(source_tokenized)
        source_tokenized.extend([self.tokenizer.token_to_id("[PAD]")] * padding_length)
        source_attention_mask.extend([0] * padding_length)

        padding_length = self.max_len - len(target_tokenized)
        target_tokenized.extend([self.tokenizer.token_to_id("[PAD]")] * padding_length)

        return {
            'input_ids': torch.tensor(source_tokenized, dtype=torch.long),
            'attention_mask': torch.tensor(source_attention_mask, dtype=torch.long),
            'labels': torch.tensor(target_tokenized, dtype=torch.long)
        }

"""
def train_model(model, tokenizer, device, loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
"""

def train_model(model, train_loader, val_loader, optimizer, device, epochs=100, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    no_improve_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train loss: {avg_train_loss}, Val loss: {avg_val_loss}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epoch = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print(f"No improvement in validation loss for {patience} consecutive epochs. Stopping early.")
                break



# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda for using gpu

# Load tokenizer customized for dataset 
tokenizer_path = os.path.join(os.getcwd(),'..','models','tokenizer','trained_tokenizer.json')  # Update this path
tokenizer = Tokenizer.from_file(tokenizer_path)
# Load the T5 model configuration
config = T5Config.from_pretrained('t5-small')
# Then initialize the T5 model from the configuration
model = T5ForConditionalGeneration(config).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

"""
# Dataset preparation
train_dataset = CustomDataset(tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=32)
"""

# Loading Dataset
full_dataset = CustomDataset(tokenizer, max_len=128)

# Splitting the dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Creating DataLoader for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Training loop
#train_model(model, tokenizer, device, train_loader, optimizer, 200)
train_model(model, train_loader, val_loader, optimizer, device, epochs=100, patience=10)


# Save the model and tokenizer
model_save_directory = os.path.join(os.getcwd(),'..','models','seq2seq')
tokenizer_save_directory = os.path.join(os.getcwd(),'..','models','seq2seq','tokenizer.json')

model.save_pretrained(model_save_directory)
tokenizer.save(tokenizer_save_directory)
