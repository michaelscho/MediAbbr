import torch
from transformers import T5ForConditionalGeneration
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

def load_model_and_custom_tokenizer(model_path, tokenizer_file_path):
    """
    Load the model and a custom tokenizer from specified paths.
    """
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # Load the custom tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_file_path)
    # Wrap the tokenizer in the transformers interface
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    return model, tokenizer

def expand_abbreviation(model, tokenizer, input_text, device='cpu'):
    """
    Expand the given abbreviation using the loaded model and tokenizer.
    """
    model.to(device)
    model.eval()
    
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Generate the expanded text
    output_ids = model.generate(input_ids, max_length=512)
    
    # Decode and return the expanded text
    expanded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return expanded_text

if __name__ == "__main__":
    # Adjust these paths to your model and custom tokenizer
    model_path = '/home/michael/git/MediAbbr/models/seq2seq/'
    tokenizer_path = '/home/michael/git/MediAbbr/models/tokenizer/trained_tokenizer.json'

    # Load the model and custom tokenizer
    model, tokenizer = load_model_and_custom_tokenizer(model_path, tokenizer_path)

    # Input text to expand
    input_text = "firmis quia cena dn̅i et parasceue et sabbatum ad illos"

    # overfitting!!!

    # Expand the abbreviation
    expanded_text = expand_abbreviation(model, tokenizer, input_text)
    
    print(f"Original: {input_text}")
    print(f"Expanded: {expanded_text}")


