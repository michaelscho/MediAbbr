import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import os

# Check dataset
""" 
Data should be separated by ; with abbreviations on the left and expansions on the right.
Length should be adjusted to max_length. Byt5 small can handle 512, larger up to 1024, 
but 256 prooved to be suitable for tasks. Length should vary across dataset.
"""

def check_data():
    with open("shuffled_output.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if ";" not in line:
                print(f"Issue at line {i + 1}: {line.strip()}")
            else:
                pass
                #input_text, output_text = line.strip().split(";")
                #print(f"Input: {input_text}, Output: {output_text}")


# Load and process data
def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            #print(line)
            input_text, output_text = line.strip().split(";")
            data.append({"input_text": input_text, "output_text": output_text})
    #print(data)
    return pd.DataFrame(data)

# Preprocessing
def preprocess_function(examples, tokenizer, max_input_length=256, max_output_length=256):
    """Tokenize the input and output sequences. 
    Important: Make sure input lenght < outputlength in data, as abbreviations are smaller. """

    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        examples["output_text"],
        max_length=max_output_length,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    print(eval_pred) 
    return {}

# Training
def train_model(data_path, model_name, output_dir, num_train_epochs=10):
    # Load and preprocess data
    df = load_data(data_path)
    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Split dataset
    split = dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    test_dataset = split["test"]

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Define training arguments (TODO: Move outside as constants)
    training_args = TrainingArguments(
        output_dir=os.path.join(os.getcwd(), "..", "models", "byt5-small-finetuned"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,  # LR must be adjusted to batch and size of dataset
        per_device_train_batch_size=8,  # When scaling max_length, this needs to be smaller on my GPU (RTX3060)
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=200,
        save_total_limit=2,
        report_to="none",
    )


    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    """ Switch to checkpoint if training is interupted """
    #checkpoint_path = "./byt5-small-finetuned/checkpoint-11565"  # Replace with your checkpoint path
    #trainer.train(resume_from_checkpoint=checkpoint_path)
    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training completed and model saved.")

def generate_with_beam_search(input_text, model_path, num_beams=5):
    """
    Generate text using beam search decoding.
    num_beams of 5 was suitable, 
    repetition_penalty helps reducing repetitions, 1.5 was suitable
    """


    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate output using beam search
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=num_beams,
        early_stopping=True,
        repetition_penalty=1.5
    )

    # Decode the generated tokens into text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Main Function
def main():

    check_data()

    # Paths and settings
    data_path = os.path.join(os.getcwd(), "..", "data", "input", "shuffled_output.txt")
    model_name = "google/byt5-small"
    output_dir = os.path.join(os.getcwd(), "..", "models", "byt5-small-finetuned")

    # Train the model
    print("Starting training...")
    train_model(data_path, model_name, output_dir, num_train_epochs=10) # 10 epochs was minimum for good results

    # Test the model
    print("Testing the model...")
    test_input = "ep\u0305s non licet sed etiam humanā gr̅am"
    output = generate_with_beam_search(test_input, output_dir, num_beams=5)
    # "episcopus non licet sed etiam humanam gratiam"
    print(f"Input: {test_input}\nOutput: {output}")

# Test the model
    print("Testing the model...")
    test_input = "Ex decr̅ Felici om̄ib fr̅ib missis"
    output = generate_with_beam_search(test_input, output_dir, num_beams=5)
    # "Ex decretis Felici omnibus factibus missis"
    print(f"Input: {test_input}\nOutput: {output}")


    test_input = "ouibus non repraehendendos quod absit"
    output = generate_with_beam_search(test_input, output_dir, num_beams=5)
    # "ouibus non repraehendendos quod absit"
    print(f"Input: {test_input}\nOutput: {output}")

    test_input = "Qđ non licet"
    output = generate_with_beam_search(test_input, output_dir, num_beams=5)
    # "Quod non licet"
    print(f"Input: {test_input}\nOutput: {output}")

    test_input = "Et tamen diis suis ista non tribuunt quoꝝ cultū ideo reqͥrunt ne ista uel ̄inora patiantur cum ea maiora ꝑtulerint a quib antea colebant"
    output = generate_with_beam_search(test_input, output_dir, num_beams=5)
    # "Et tamen diis suis ista non tribuunt quorum cultum ideo requirunt ne ista uel minora patiantur cum ea maiora pertulerint a quibus antea colebantur"
    print(f"Input: {test_input}\nOutput: {output}")



if __name__ == "__main__":
    main()
