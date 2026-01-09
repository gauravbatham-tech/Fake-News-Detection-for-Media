📰 Fake News Detection Using Machine Learning & BERT

This project builds a system that can classify news as real or fake using machine learning and deep learning.
It was developed as part of my internship to understand NLP preprocessing, ML pipelines, and how transformer-based models like BERT improve semantic understanding.

📌 Project Overview

Fake news is a growing issue, and detecting it automatically is an important NLP challenge.
This project focuses on building a hybrid system using:

Traditional ML (TF-IDF + Logistic Regression)

Deep Learning (BERT transformer model)

The notebook covers the full workflow from data loading to model evaluation and prediction.

📂 Features

This project includes:

Dataset merging and labeling

Text preprocessing and cleaning

Train-test split

TF-IDF feature extraction

Logistic Regression classifier

BERT sequence classification

Evaluation metrics and confusion matrix

Automatic prediction interface (no user input required)

Organized, internship-ready notebook with clear structure

🛠 Tech Stack

Language: Python
Libraries:
NumPy, Pandas, Matplotlib, Seaborn,
Scikit-learn, NLTK,
Transformers (HuggingFace), PyTorch
Tools: Google Colab / Jupyter Notebook
Dataset: Kaggle Fake & True News Dataset
(Consisting of True.csv and Fake.csv)

DATASET: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

🚀 Workflow
1. Data Loading and Merging

Loaded True.csv and Fake.csv

Assigned labels:

0 = Real

1 = Fake

Combined both files into one unified dataset

2. Text Preprocessing

Converted text to lowercase

Removed digits and punctuation

Removed stopwords

Created a new column clean_text

3. Train–Test Split

Stratified split for balanced evaluation

Clean and reproducible setup

4. Machine Learning Model (TF-IDF + Logistic Regression)

Extracted features using TF-IDF

Trained Logistic Regression

Evaluated accuracy, precision, recall, F1

Added confusion matrix heatmap

5. BERT (Deep Learning Model)

Used BERT tokenizer

Built a custom PyTorch dataset class

Loaded bert-base-uncased

Trained on a small subset for faster runtime

Evaluated with accuracy, precision, recall

6. Automatic Prediction Interface

Created a clean prediction function

Added three sample news statements

Automatically prints predictions without user input

📊 Results Summary
Model	Strength	Notes
TF-IDF + Logistic Regression	Fast, high accuracy	Excellent baseline classifier
BERT Transformer Model	Deeper context understanding	Better on semantic meaning

Both models perform well, but BERT captures context better, while LR is faster and simpler.

🏁 Conclusion

This project demonstrates how machine learning and deep learning can detect fake news effectively.
It highlights the importance of:

Text preprocessing

TF-IDF vectorization

Transformer models

Evaluation metrics

