# utils.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def compute_metrics(true_labels, predicted_labels):
    # Print a full classification report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))
    
    # Compute individual metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="weighted"),
        "recall": recall_score(true_labels, predicted_labels, average="weighted"),
        "f1_score": f1_score(true_labels, predicted_labels, average="weighted"),
    }
    
    # Print individual metrics
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")
    
    return metrics
