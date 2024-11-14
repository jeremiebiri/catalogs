import os
import torch
import pandas as pd   
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer
)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name, num_labels=2):
    """
    Load model and tokenizer for a given model name.
    Supports both sequence classification and seq2seq models.
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if "t5" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            from_flax=not os.path.exists(os.path.join(model_name, "pytorch_model.bin"))
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            from_flax=not os.path.exists(os.path.join(model_name, "pytorch_model.bin"))
        )

    model.to(device)
    return model, tokenizer

def tokenize_data(data, tokenizer, max_length=128):
    from datasets import Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    return dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_length), batched=True)

def get_training_arguments(output_dir="./results", epochs=3, batch_size=8):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        save_total_limit=1
    )

class CustomTrainer(Trainer):
    def _save(self, output_dir: str, state_dict=None):
        for name, param in self.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.contiguous()
        super()._save(output_dir, state_dict)

def create_predictions_catalog(trainer, test_dataset, device, model_name, num_labels, epochs, batch_size):
    predictions = trainer.predict(test_dataset)
    preds = torch.tensor(predictions.predictions).to(device).argmax(dim=1)
    
    catalog = {
        "use_case": "Text Classification",
        "model": model_name,
        "model_parameters": {
            "num_labels": num_labels,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "results": []
    }

    for i in range(len(test_dataset)):
        note = test_dataset["text"][i]
        true_label = test_dataset["label"][i]
        predicted_label = preds[i].item()
        confidence_score = float(predictions.predictions[i].max())  
        catalog["results"].append({
            "note": note,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence_score": confidence_score,
        })
    
    return preds, catalog

def compute_metrics(true_labels, predicted_labels):
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))
    
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="weighted"),
        "recall": recall_score(true_labels, predicted_labels, average="weighted"),
        "f1_score": f1_score(true_labels, predicted_labels, average="weighted"),
    }
    
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")
    
    return metrics

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

 

















# import os
# import torch
# import pandas as pd   
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForSequenceClassification, 
#     AutoModelForSeq2SeqLM, 
#     TrainingArguments, 
#     Trainer
# )


# def get_device():
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_model_and_tokenizer(model_name, num_labels=2):
#     """
#     Load model and tokenizer for a given model name.
#     Supports both sequence classification and seq2seq models.
#     """
#     device = get_device()
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
    
#     # Check if the model is a T5 variant for seq2seq, otherwise default to classification
#     if "t5" in model_name.lower():
#         model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name, 
#             from_flax=not os.path.exists(os.path.join(model_name, "pytorch_model.bin"))
#         )
#     else:
#         model = AutoModelForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=num_labels,
#             from_flax=not os.path.exists(os.path.join(model_name, "pytorch_model.bin"))
#         )

#     model.to(device)
#     return model, tokenizer

# def tokenize_data(data, tokenizer, max_length=128):
#     """
#     Tokenizes a list of dictionaries containing 'text' and 'label' fields.
#     Returns a tokenized dataset ready for training or evaluation.
#     """
#     from datasets import Dataset
#     df = pd.DataFrame(data)
#     dataset = Dataset.from_pandas(df)
#     return dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_length), batched=True)

# def get_training_arguments(output_dir="./results", epochs=3, batch_size=8):
#     """
#     Returns training arguments for the Trainer class.
#     """
#     return TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         save_steps=500,
#         save_total_limit=1
#     )

# class CustomTrainer(Trainer):
#     """
#     Custom Trainer to handle any issues with non-contiguous tensors.
#     """
#     def _save(self, output_dir: str, state_dict=None):
#         for name, param in self.model.named_parameters():
#             if not param.is_contiguous():
#                 param.data = param.contiguous()
#         super()._save(output_dir, state_dict)

# def create_predictions_catalog(trainer, test_dataset, device, model_name, num_labels, epochs, batch_size):
#     """
#     Generate predictions for the test dataset and create a catalog of results.
#     """
#     predictions = trainer.predict(test_dataset)
#     preds = torch.tensor(predictions.predictions).to(device).argmax(dim=1)
    
#     # Create the catalog
#     catalog = {
#         "use_case": "Text Classification",
#         "model": model_name,
#         "model_parameters": {
#             "num_labels": num_labels,
#             "epochs": epochs,
#             "batch_size": batch_size,
#         },
#         "results": []
#     }

#     # Populate catalog with predictions
#     for i in range(len(test_dataset)):
#         note = test_dataset["text"][i]
#         true_label = test_dataset["label"][i]
#         predicted_label = preds[i].item()
#         confidence_score = float(predictions.predictions[i].max())  # Convert to Python float
#         catalog["results"].append({
#             "note": note,
#             "true_label": true_label,
#             "predicted_label": predicted_label,
#             "confidence_score": confidence_score,
#         })
    
#     return preds, catalog
# # utils.py

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# def compute_metrics(true_labels, predicted_labels):
#     # Print a full classification report
#     print("Classification Report:")
#     print(classification_report(true_labels, predicted_labels))
    
#     # Compute individual metrics
#     metrics = {
#         "accuracy": accuracy_score(true_labels, predicted_labels),
#         "precision": precision_score(true_labels, predicted_labels, average="weighted"),
#         "recall": recall_score(true_labels, predicted_labels, average="weighted"),
#         "f1_score": f1_score(true_labels, predicted_labels, average="weighted"),
#     }
    
#     # Print individual metrics
#     for metric, value in metrics.items():
#         print(f"{metric.capitalize()}: {value:.2f}")
    
#     return metrics
