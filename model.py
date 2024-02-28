import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the dataset
file_path = 'src/datasets/Truth_Seeker_Model_Dataset.csv'
df = pd.read_csv(file_path)

# Splitting the dataframe based on the 'target' column
df_true = df[df['target'] == True]
df_false = df[df['target'] == False]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text
texts = df['tweet'].tolist()  # Use the Tweet column for classification

# Tokenize and prepare for BERT
input_ids = []
attention_masks = []

for text in texts:
    encoded_dict = tokenizer.encode_plus(
                        text,                      # Text to encode
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attention masks
                        return_tensors = 'pt',     # Return pytorch tensors
                )
    
    # Add the encoded sentence to the list    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding)
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['BinaryNumTarget'].tolist())  # Assuming 'BinaryNumTarget' is your target variable

# You now have `input_ids`, `attention_masks`, and `labels` for your dataset
