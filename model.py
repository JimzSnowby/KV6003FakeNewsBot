import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

# Load the dataset
file_path = 'src/datasets/Truth_Seeker_Model_Dataset.csv'
df = pd.read_csv(file_path)

missing_values_count = df.isnull().sum() # we get the number of missing data points per column
print("Number of missing data points per column:\n")
print (missing_values_count)
df.dropna(inplace=True) #drop the empty rows

train_df, sub_df = train_test_split(df, stratify=df.BinaryNumTarget.values, 
                                                random_state=42, 
                                                test_size=0.2, shuffle=True)

validation_df, test_df = train_test_split(sub_df, stratify=sub_df.BinaryNumTarget.values, 
                                                random_state=42, 
                                                test_size=0.25, shuffle=True)

train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# We check the number of examples after split
print("Train data: {} \n".format(train_df.shape))
print("Validation data: {} \n".format(validation_df.shape))
print("Test data: {} \n".format(test_df.shape))


