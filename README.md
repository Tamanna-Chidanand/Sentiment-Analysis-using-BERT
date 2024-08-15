# Sentiment-Analysis-using-BERT
This repository implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers) to classify text sentiments into categories such as positive, negative, and neutral.
Features
Pretrained BERT Model: Utilizes Hugging Face's Transformers library to leverage a pretrained BERT model for improved accuracy and performance.
Data Processing: Includes scripts for cleaning and preprocessing text data to prepare it for model training and evaluation.
Training and Evaluation: Implements training routines to fine-tune the BERT model on a sentiment analysis dataset and evaluate its performance using accuracy, precision, recall, and F1 score metrics.
User-friendly Interface: Provides a simple interface to input text for real-time sentiment prediction.
Requirements
Python 3.x
PyTorch
Transformers
Pandas
Scikit-learn
Getting Started
Clone the repository
Install the required packages: pip install -r requirements.txt
Run the training script: python train.py
Use the inference script to predict sentiments: python predict.py
Usage
You can input any text, and the model will output the predicted sentiment. The trained model can also be fine-tuned with your own dataset.
