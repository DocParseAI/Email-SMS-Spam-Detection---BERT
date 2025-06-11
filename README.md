# SMS/Email Spam Detection - BERT Trasformer and Naive Bayes Classifier

This repository contains two Jupyter Notebooks demonstrating SMS/email spam detection using two different approaches: a traditional Naive Bayes classifier and a fine-tuned BERT model. The BERT model is available on Hugging Face: [SGHOSH1999/bert-email-spam-classifier_tuned](https://huggingface.co/SGHOSH1999/bert-email-spam-classifier_tuned).

---

## Notebooks Overview

### 1. Naive Bayes Spam Classifier (`naive_bayes_spam_classifier.ipynb`)
- **Purpose:** Introduces a classic machine learning approach for spam detection using the Multinomial Naive Bayes algorithm.
- **Workflow:**
    - Data preprocessing (cleaning, tokenization, vectorization with TF-IDF)
    - Model training and evaluation
- **Best For:** Quick, interpretable baseline and comparison with deep learning models.

### 2. BERT Spam Classifier (`spam-sms-classification-bert.ipynb`)
- **Purpose:** Demonstrates a modern deep learning approach using BERT (Bidirectional Encoder Representations from Transformers) for binary classification (spam vs. ham).
- **Workflow:**
    - Data preprocessing and tokenization using Hugging Face Transformers
    - Fine-tuning `bert-base-uncased` on the SMS Spam Collection Dataset
    - Model evaluation and export
- **Training Details:**
    - **Batch Size:** 16
    - **Learning Rate:** 5e-5
    - **Epochs:** 3
- **Performance:**
    - **Loss:** 0.012
    - **Accuracy:** 0.98

---

## Dataset

- **Name:** SMS Spam Collection Dataset
- **Description:** 5,574 SMS messages labeled as 'ham' (legitimate) or 'spam'.
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

---

## Model Details (BERT)

- **Architecture:** BERT (bert-base-uncased)
- **Task:** Binary classification (spam vs. ham)
- **Framework:** Hugging Face Transformers (PyTorch)
- **License:** MIT

### Usage

Load the model and tokenizer from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("SGHOSH1999/bert-email-spam-classifier_tuned")
model = AutoModelForSequenceClassification.from_pretrained("SGHOSH1999/bert-email-spam-classifier_tuned")
```

---

## Citation

If you use this model or code, please cite the original dataset and this repository.

---

## License

MIT License