from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline, DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np
import pandas as pd

# Load the dataset
dataset = load_dataset('csv', data_files='src/datasets/Truth_Seeker_Model_Dataset.csv')

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess(examples):
    return tokenizer(examples['tweet'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=512)

accuracy = evaluate.load('accuracy')

def calc_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

id2label = {0: 'False', 1: 'True'}
label2id = {'False': 0, 'True': 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

print('End Reached')