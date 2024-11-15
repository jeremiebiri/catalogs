import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils import (
    get_device, load_model_and_tokenizer, tokenize_data, get_training_arguments, CustomTrainer, create_predictions_catalog, compute_metrics
)

# Set up device
device = get_device()
print(f"Using device: {device}")

# Define model and data
model_name = "./Bio_ClinicalBERT"  # Path to model or Hugging Face model ID
num_labels = 3  # Number of labels for classification

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_name, num_labels=num_labels)

# Load the synthetic data
synthetic_data = pd.read_csv("synthetic_medical_data.csv")
data = synthetic_data.to_dict(orient="records")  # Convert to list of dicts for compatibility with tokenize_data

# Tokenize data
tokenized_datasets = tokenize_data(data, tokenizer, max_length=128)

# Split data into training and test sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Define training arguments (required for Trainer setup, even if not training)
training_args = get_training_arguments(output_dir="./results", epochs=3, batch_size=8)

# Set up Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
fine_tuned_model_path = "./fine_tuned_Bio_ClinicalBERT"
trainer.save_model(fine_tuned_model_path)  # Saves the model weights and configuration
tokenizer.save_pretrained(fine_tuned_model_path)  # Saves the tokenizer files
print("Fine-tuned model and tokenizer saved.")

# Reload the fine-tuned model and tokenizer for predictions
model, tokenizer = load_model_and_tokenizer(fine_tuned_model_path, num_labels=num_labels)

# Re-setup Trainer with the fine-tuned model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=test_dataset,
)

# Generate predictions
preds, catalog = create_predictions_catalog(
    trainer, test_dataset, device, model_name, num_labels, 
    training_args.num_train_epochs, training_args.per_device_train_batch_size
)

# Compute metrics
metrics = compute_metrics(test_dataset["label"], preds.cpu())

# Plot the metrics
def plot_metrics(metrics):
    """
    Plots a bar chart of the provided metrics.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(10, 5))
    plt.bar(metric_names, metric_values, color='skyblue')
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # Set y-axis limit to 1.0 for percentages
    plt.title("Model Performance Metrics")
    plt.show()

# Call plot_metrics to display the bar chart
plot_metrics(metrics)

# Print the classification report
print("Classification Report:")
print(classification_report(test_dataset["label"], preds.cpu()))  # Move preds to CPU for reporting
