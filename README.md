# catalogs
from zipfile import ZipFile

# Create the README content
readme_content = """
# Medical Text Classification with BERT Models

This repository contains code for classifying medical text data using transformer models, such as Bio_ClinicalBERT, BioBERT, and ClinicalT5. You can load these models either from a local directory or directly from Hugging Face's Model Hub by specifying a model ID and using an API key if needed.

## Prerequisites

Make sure you have the following tools installed:

- **Python 3.8+**
- **Git**: For cloning the repository.
- **CUDA** (Optional): If you want to use GPU for acceleration and have a compatible NVIDIA GPU.

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jeremiebiri/catalogs.git
cd catalogs

#Create python env (Windows)
python -m venv pythorch
pythorch\\Scripts\\activate
# Env (Debian)
python3 -m venv pythorch
source pythorch/bin/activate

#Install requirement
pip install -r requirements.txt

##Project Setup and Execution Guide
This guide will walk you through setting up and running the project locally. The instructions include how to clone models directly from Hugging Face and run them without accessing the Hugging Face API.

#Setting Up the Model Locally
##Step 1: Clone the Model from Hugging Face Locally
If you are using the Bio_ClinicalBERT or another Hugging Face model, follow these steps to clone it directly:

git clone https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT ./Bio_ClinicalBERT

 