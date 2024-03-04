import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def load_split(filename):
    df = pd.read_csv(filename, dtype={'BinaryNumTarget' : 'Int32'})
    print(df.head())

    # Check class distribution
    print(df['BinaryNumTarget'].value_counts(normalize = True))

    # Split train dataset into train, validation and test sets
    train_text, temp_text, train_labels, temp_labels = train_test_split(df['tweet'], df['BinaryNumTarget'],
                                                                        random_state=2018,
                                                                        test_size=0.3,
                                                                        stratify=df['BinaryNumTarget'])

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                        random_state=2018,
                                                                        test_size=0.5,
                                                                        stratify=temp_labels)

    # Get length of all the messages in the train set
    seq_len = [len(i.split()) for i in train_text]

    pd.Series(seq_len).hist(bins = 30)
    plt.show()
    
    return train_text, temp_text, train_labels, temp_labels, val_text, test_text, val_labels, test_labels