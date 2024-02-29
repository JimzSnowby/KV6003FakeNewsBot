import pandas as pd

def preprocess (df):
    # drop columns not required for the model
    df.drop(['author', 'target', 'manual_keywords', '5_label_majority_answer', '3_label_majority_answer'], axis=1)
    
    
    
    return