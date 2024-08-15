import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset
file_path = 'C:/Users/tamch/Documents/LLM_miniproject/IMDBDataset.csv'
df = pd.read_csv(file_path)

# Function to map labels
def map_sentiment_label(label):
    if label == 'negative':
        return 0
    elif label == 'neutral':
        return 1
    elif label == 'positive':
        return 2
    else:
        return None  # Handle invalid or NaN values

# Apply label mapping, handling NaN values
df['sentiment'] = df['sentiment'].apply(lambda x: map_sentiment_label(x) if pd.notnull(x) else x)

# Drop rows with NaN values in 'sentiment' column
df.dropna(subset=['sentiment'], inplace=True)

# Define a custom dataset class
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

# Preprocess the data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 128  # Adjust max length as needed

# Use a smaller subset of the data for faster training
df = df.sample(1000, random_state=42)  # Use only 1000 samples

# Split data into training and validation sets
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)

# Create datasets and data loaders
train_dataset = MovieReviewDataset(
    reviews=train_data['review'].values,
    labels=train_data['sentiment'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
val_dataset = MovieReviewDataset(
    reviews=val_data['review'].values,
    labels=val_data['sentiment'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained DistilBERT model for sequence classification with 3 classes
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EPOCHS = 3

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Train loss: {train_loss:.4f} accuracy: {train_acc:.4f}")

    val_acc, val_loss = eval_model(model, val_loader, device)
    print(f"Val   loss: {val_loss:.4f} accuracy: {val_acc:.4f}")

# Function to predict sentiment of a single review
def predict_sentiment(review):
    encoded_review = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

    sentiment = "negative" if prediction == 0 else ("neutral" if prediction == 1 else "positive")
    return sentiment

# Test the function

print(predict_sentiment("This movie was excellent!"))  # Should output 'positive'
print(predict_sentiment("I am not a fan of it"))  # Should output 'negative'
#print(predict_sentiment("It was not good but not bad"))      

statement=input("Enter your review: ")
print(predict_sentiment(statement))



