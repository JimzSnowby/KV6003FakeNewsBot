from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
import pandas as pd


# Load the dataset and split it into training and testing sets
dataset = load_dataset('csv', data_files='src/datasets/Truth_Seeker_Model_Dataset.csv')
train_test_split = dataset['train'].train_test_split(test_size=0.2)
dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

# Load Tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Preprocess to ensure the data is in the correct format for the model
def preprocess(examples):
    return tokenizer(examples['tweet'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=512)

accuracy = evaluate.load('accuracy')

# Define the function to calculate the performance of the model
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

id2label = {0: 'False', 1: 'True'}
label2id = {'False': 0, 'True': 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir='src/models/FN_Truth_Seeker_Model',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()