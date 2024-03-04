from transformers import BertTokenizerFast
import torch

def tokenizer(train_text, val_text, test_text):
    
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = 25,
        padding='max_length',
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = 25,
        padding='max_length',
        truncation=True
    )

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = 25,
        padding='max_length',
        truncation=True
    )
    
    return tokens_train, tokens_val, tokens_test