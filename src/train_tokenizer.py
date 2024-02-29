from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

""" Train custom tokenizer using BPE and subtokens to account for brevigraphs and expansion"""

# Initialize BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Split split the text into words
tokenizer.pre_tokenizer = Whitespace()

# Initialize BPE Trainer model with basic parameters
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Path to your dataset
file_paths = [os.path.join(os.getcwd(),'..','data','training','tokenizer','seq2seq_gt.txt')]

# Train the tokenizer
tokenizer.train(files=file_paths, trainer=trainer)

# Save the tokenizer
tokenizer.save(os.path.join(os.getcwd(),'..','models','tokenizer','trained_tokenizer.json'))
