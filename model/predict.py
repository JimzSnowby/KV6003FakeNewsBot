import torch
import numpy as np
from sklearn.metrics import classification_report

# Assuming your Model class is defined in model.py
from model import Model

class Predict:
    def __init__(self, model_path='saved_weights.pt'):
        # Initialize the Model class
        self.model_instance = Model()
        
        # Load the trained model weights and set model to evaluation mode
        self.model_instance.model.load_state_dict(torch.load(model_path))
        self.model_instance.model.eval()
        
        # Use the same device as the Model class
        self.device = self.model_instance.device
        print("INIT")

    def get_predictions(self, test_seq, test_mask, test_y):
        # Use the same device as the Model class
        test_seq = test_seq.to(self.device)
        test_mask = test_mask.to(self.device)

        with torch.no_grad():
            preds = self.model_instance.model(test_seq, test_mask)
            preds = preds.detach().cpu().numpy()

        # Convert predictions to labels
        preds = np.argmax(preds, axis=1)
        
        # Print classification report
        print(classification_report(test_y, preds))
        print("PREDICT")