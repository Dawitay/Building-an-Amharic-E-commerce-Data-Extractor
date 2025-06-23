import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import numpy as np

# Define paths
data_path = '../data/preprocessed/labeled_telegram_product_price_location.txt'

# Load and preprocess data (Custom dataset loading for your task)
def load_data():
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    sentence = []
    for line in lines:
        if line.strip():  # Non-empty line
            token, label = line.strip().split('\t')
            sentence.append((token, label))
        else:
            if sentence:
                data.append(sentence)
                sentence = []
    
    return data

# Load the labeled dataset
data = load_data()

# Convert to a Hugging Face Dataset format (for easy integration with the trainer)
dataset = Dataset.from_dict({"tokens": [item[0] for sublist in data for item in sublist],
                             "labels": [item[1] for sublist in data for item in sublist]})

# Initialize the tokenizer
model_name = "xlm-roberta-base"  # Example model, can be replaced with other models
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['tokens'], padding="max_length", truncation=True, is_split_into_words=True)

dataset = dataset.map(tokenize_function, batched=True)

# Prepare the data for Trainer
train_dataset = dataset.shuffle(seed=42).select([i for i in range(100)])  # You can adjust this for real training data
val_dataset = dataset.shuffle(seed=42).select([i for i in range(100, 120)])

# Define model configurations for different models
models = {
    "XLM-Roberta": "xlm-roberta-base",
    "DistilBERT": "distilbert-base-uncased",
    "mBERT": "bert-base-multilingual-cased",
}

# Function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = labels
    return classification_report(true_labels, predictions, output_dict=True)

# Fine-tune and evaluate each model
for model_name, model_path in models.items():
    print(f"Fine-tuning {model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(dataset.features['labels'].unique()))

    training_args = TrainingArguments(
        output_dir=f'./results/{model_name}',          # where to store the final model
        num_train_epochs=3,                           # number of training epochs
        per_device_train_batch_size=8,                # batch size for training
        per_device_eval_batch_size=8,                 # batch size for evaluation
        warmup_steps=500,                             # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                            # strength of weight decay
        logging_dir=f'./logs/{model_name}',           # directory for storing logs
        evaluation_strategy="epoch",                  # evaluation strategy to use
        logging_strategy="epoch",                     # logging strategy to use
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,                     # the model to be trained
        args=training_args,              # training arguments
        train_dataset=train_dataset,     # training dataset
        eval_dataset=val_dataset,        # evaluation dataset
        compute_metrics=compute_metrics  # function to compute metrics during evaluation
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    print(f"Evaluating {model_name}...")
    results = trainer.evaluate()

    # Display the evaluation results
    print(f"{model_name} Evaluation Results: {results}")

    # Save the model
    model.save_pretrained(f'./models/{model_name}')

# Compare and select the best model based on the evaluation metrics
# This part depends on which metric you prioritize, e.g., accuracy, F1 score, etc.
# Here, the evaluation results are stored in `results`, which you can use for comparison.
