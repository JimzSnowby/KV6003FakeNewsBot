import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel
from pathlib import Path

from tokenizer import tokenizer
from load_split import load_split
from bert_arch import BERT_Arch
from train_model import train_model
from evaluate import evaluate

class Model:
    # Specify GPU
    device = torch.device("cuda")
    
    filename = "src/datasets/Truth_Seeker_Model_Dataset.csv"
    train_text, temp_text, train_labels, temp_labels, val_text, test_text, val_labels, test_labels = load_split(filename)
    
    # Import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    # Initialize Tokenizer with the datasets
    tokens_train, tokens_val, tokens_test = tokenizer(train_text, val_text, test_text)
    
    # convert lists to tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())
    
    #define a batch size
    batch_size = 32

    # wrap tensors and sample data during training
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False
        
    # passes the pre-trained BERT to our define architecture and push to GPU
    model = BERT_Arch(bert)
    model = model.to(device)

    # Define optimizer and compute class weights
    optimizer = torch.optim.AdamW(model.parameters(),lr = 1e-5) 
    class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels), y = train_labels)
    print("Class Weights:",class_weights)

    # path to the weights produced by training
    weights_path = Path('model/saved_weights.pt')

    # Check if the weights file exists
    if not weights_path.exists():
        print("TRAINING")
        # convert list of class weights to a tensor
        weights= torch.tensor(class_weights,dtype=torch.float)

        # push to GPU
        weights = weights.to(device)
        # define the loss function
        cross_entropy  = nn.NLLLoss(weight=weights) 
        
        avg_loss, total_preds = train_model(model, device, train_dataloader, optimizer, cross_entropy)
        avg_loss, total_preds = evaluate(model, device, val_dataloader, cross_entropy)
        
        # number of training epochs
        epochs = 3
        # set initial loss to infinite
        best_valid_loss = float('inf')

        # empty lists to store training and validation loss of each epoch
        train_losses=[]
        valid_losses=[]

        # for each epoch
        for epoch in range(epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
            
            # train and evaluate model, doesn't return total_preds
            train_loss, _ = train_model(model, device, train_dataloader, optimizer, cross_entropy)
            valid_loss, _ = evaluate(model, device, val_dataloader, cross_entropy)
            
            # save the best model to a file
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'saved_weights.pt')
            
            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')
    else:
        print("PREDICTING")
        # Load the weights from the file
        path = 'model/saved_weights.pt'
        model.load_state_dict(torch.load(path))
        # Process in batches
        batch_size = 10  # Adjust based on GPU memory
        preds_list = []
        with torch.no_grad():
            for i in range(0, len(test_seq), batch_size):
                batch_seq = test_seq[i:i+batch_size].to(device)
                batch_mask = test_mask[i:i+batch_size].to(device)
                batch_preds = model(batch_seq, batch_mask)
                batch_preds = batch_preds.detach().cpu().numpy()
                preds_list.append(batch_preds)
                print("batch: ", i)
                
        preds = np.concatenate(preds_list, axis=0)
        preds = np.argmax(preds, axis=1)
        print(classification_report(test_y, preds))
    
    print("complete")
