import random
import pandas as pd

# Define the categories and labels
categories = {
    0: "Diabetes",
    1: "Hypertension",
    2: "Asthma"
}

# Define example symptoms or notes for each category
sample_texts = {
    0: [
        "Patient has a long history of type 2 diabetes with recent complications.",
        "Blood sugar levels are consistently high despite medication.",
        "Diabetes management needs improvement due to poor glycemic control.",
        "Frequent urination and excessive thirst reported; possible diabetes symptoms.",
        "Recent A1C test shows elevated levels, patient needs diabetic counseling."
    ],
    1: [
        "Hypertension is being managed with medication, but blood pressure remains high.",
        "Patient reports occasional headaches and dizziness, indicative of high blood pressure.",
        "Blood pressure readings have been consistently above normal range.",
        "Medication for hypertension adjusted due to side effects.",
        "Elevated blood pressure observed during recent check-up."
    ],
    2: [
        "Asthma symptoms worsening, especially during cold weather.",
        "Frequent shortness of breath and wheezing reported by patient.",
        "Asthma management plan reviewed with patient; inhaler prescribed.",
        "Patient reports increased coughing and difficulty breathing at night.",
        "Respiratory issues linked to asthma, particularly during exercise."
    ]
}

# Generate synthetic data
def generate_synthetic_data(num_samples=100):
    data = []
    for _ in range(num_samples):
        label = random.choice(list(categories.keys()))
        text = random.choice(sample_texts[label])
        data.append({"text": text, "label": label})
    return pd.DataFrame(data)

# Generate 100 samples of synthetic medical data
synthetic_data = generate_synthetic_data(num_samples=100)

# Save synthetic data to CSV (optional, if you want to keep a record)
synthetic_data.to_csv("synthetic_medical_data.csv", index=False)

# Display the first few rows of the synthetic data
print(synthetic_data.head())
