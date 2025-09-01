# Tweet-Classification-using-LLM
Fine-tune transformer models like DeBERTa-v3 for age classification of tweets. Includes preprocessing, dataset handling, training with Hugging Face Trainer, and evaluation with accuracy, F1, and confusion matrix. Easily extendable to other NLP tasks

Age Classification with Transformers
This repository contains a PyTorch + Hugging Face Transformers pipeline for age classification of tweets. The project fine-tunes transformer models like DeBERTa-v3 for sequence classification, with preprocessing, dataset handling, training, and evaluation.

Features

Preprocessing tweets (replace @USER and URLs).
Train/validation/test split with shuffling.
Hugging Face Trainer for fine-tuning transformers.
Supports DeBERTa-v3, BERT, XLM-RoBERTa, and others.
Evaluation with accuracy, macro F1-score, and confusion matrix visualization.

Dataset
This project expects the following input CSV files:

age_tain_2024.csv
age_2024_dev.csv
og_test1.csv
Each file should contain:
tweet (text field)
task1_hardlabel (classification label)

Requirements

torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm

Training
The model will:
Tokenize tweets with DeBERTa-v3 tokenizer.
Train for 3 epochs with batch size 8.
Save checkpoints in ./output.

Evaluation

After training, the script computes:
Validation Accuracy
Macro F1 Score
Confusion Matrix


Code
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer  
from transformers import BertTokenizer, BertModel
import os.path
from os import path
import pandas as pd
import numpy as np
import random
import  matplotlib. pyplot  as  plt
from tqdm import tqdm

import torch
import torch.optim as optim
import  torch. nn. functional  as  F
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer  
import os.path
from os import path 

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from sklearn.preprocessing import LabelEncoder
task1_encoder = LabelEncoder()

import re
def simple_preprocess(text):
  """
  pass the tweet data as a series. do not use apply function
  only preprocesses for replacing @USER and URLS
  """
  # print("i am preprocessing")
  URL_RE = re.compile(r"https?:\/\/[\w\.\/\?\=\d&#%_:/-]+")
  HANDLE_RE = re.compile(r"@\w+")
  tweets = []
  for t in text:
    t = HANDLE_RE.sub("@USER", t)
    t = URL_RE.sub("HTTPURL", t)
    tweets.append(t)
  return tweets



#load given data
import pandas as pd
load_train1 = pd.read_csv("/kaggle/input/dvgchdbh/age_tain_2024.csv")
load_dev1 = pd.read_csv("/kaggle/input/dvgchdbh/age_2024_dev.csv")
og_test1 = pd.read_csv("/kaggle/input/dvgchdbh/og_test1.csv")

all_task1_hard_labels = pd.concat([load_train1["task1_hardlabel"],load_dev1["task1_hardlabel"]])
train1_df = (load_train1)

# print(train1_df.columns)
train1_df = train1_df[["tweet","task1_hardlabel"]].dropna()
#train1_df = train1_df[train1_df['task1_hardlabel'] != 2]
train1_df["tweet"] = simple_preprocess(train1_df["tweet"])

dev1_df = (load_dev1)
dev1_df = dev1_df[["tweet","task1_hardlabel"]].dropna()
#dev1_df = dev1_df[dev1_df['task1_hardlabel'] != 2]
dev1_df["tweet"] = simple_preprocess(dev1_df["tweet"])

test1_df = og_test1
test1_df = test1_df[["tweet"]]
# test1_df = test1_df[test1_df['hard_label'] != 2]
test1_df["tweet"] = simple_preprocess(test1_df["tweet"])

#print("train1",train1_df.shape)
#print(train1_df.head)
print("test1",test1_df.shape)
print(test1_df.head)


import pandas as pd
from sklearn.model_selection import train_test_split
# Combine train1df and test1df into a single dataframe
combined_df = pd.concat([train1_df, dev1_df], ignore_index=True)

# Shuffle the combined dataframe
combined_df_shuffled = combined_df.sample(frac=1, random_state=42)

# Split the shuffled dataframe into train, validation, and test dataframes with an 80-10-10 split
#train_df, val_df = train_test_split(combined_df_shuffled, test_size=0.15, random_state=42)
#test_df = test1_df

# Reset the indices of the dataframes
##val_df = val_df.reset_index(drop=True)
#test_df = test_df.reset_index(drop=True)

print("combined_df_shuffled",combined_df_shuffled)
#print(combined_df_shuffled.head)
train_df, val_df = train_test_split(combined_df_shuffled, test_size=0.15, random_state=42)
test_df = test1_df
# Reset the indices of the dataframes
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
#test_df
#train_df
val_df
def convert_to_dataset(df):
    df = {"text": df['tweet'].tolist(), "label": df["task1_hardlabel"].tolist()}
    dataset = Dataset.from_dict(df)
    return dataset

def convert_to_dataset_test(df):
    df = {"text": df['tweet'].tolist()}
    dataset = Dataset.from_dict(df)
    return dataset

# Convert dataframe to dataset
train_dataset = convert_to_dataset(train_df)
val_dataset = convert_to_dataset(val_df)
test_dataset = convert_to_dataset_test(test_df)
train_dataset
val_dataset
test_dataset
# Import necessary libraries

from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load the DeBERTa-v3 tokenizer and model


from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')
#from transformers import TrainingArguments

# Set parameters
MAX_LENGTH = 128

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",                           # Directory for saving the trained model
    num_train_epochs=3,                              # Number of training epochs
    per_device_train_batch_size=8,                   # Batch size per GPU/TPU core for training
    per_device_eval_batch_size=16,                   # Batch size per GPU/TPU core for evaluation
    learning_rate=2e-05,                             # Learning rate (default is 5e-5)
    weight_decay=0.004891290652279793,               # Weight decay parameter for controlling L2 regularization
    #logging_dir="./logs",                            # Directory for storing logs
    evaluation_strategy="epoch",                     # Evaluation strategy during training ("epoch" or "steps")
    save_strategy="epoch",                           # Checkpoint save strategy ("epoch" or "steps")
    logging_strategy="epoch",                        # Logging strategy during training ("epoch" or "steps")
)
def convert_to_dataset(df):
    df = {"text": df['tweet'].tolist(), "label": df["task1_hardlabel"].tolist()}
    dataset = Dataset.from_dict(df)
    return dataset

def convert_to_dataset_test(df):
    df = {"text": df['tweet'].tolist()}
    dataset = Dataset.from_dict(df)
    return dataset
train_dataset = convert_to_dataset(train_df)
val_dataset = convert_to_dataset(val_df)
test_dataset = convert_to_dataset_test(test_df)
# Create the datasets
train_encodings = tokenizer(train_dataset["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "label": train_dataset["label"]})

val_encodings = tokenizer(val_dataset["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"], "attention_mask": val_encodings["attention_mask"], "label": val_dataset["label"]})


test_encodings = tokenizer(test_dataset["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]})
def compute_f1_score(pred):
    # pred is a tuple (predictions, labels)
    predictions, labels = pred
    # Compute the F1 score
    f1 = f1_score(labels, predictions.argmax(axis=1), average='macro')
    return {"f1_score": f1}

# Define the trainer for each fold
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model for each fold
trainer.train()

from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
# Get predictions on the validation set
val_predictions = trainer.predict(val_dataset)
val_pred_labels = np.argmax(val_predictions.predictions, axis=1)
val_true_labels = val_dataset["label"]

val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
val_f1_score = f1_score(val_true_labels, val_pred_labels)

print("Validation Accuracy:", val_accuracy)
print("Validation F1 Score:", val_f1_score)
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Compute confusion matrix
cm = confusion_matrix(val_true_labels , val_pred_labels)
print("Confusion Matrix:")
print(cm)

# Visualization using Matplotlib and Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
