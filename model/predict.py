from transformers import pipeline

text = "Pier Morgan tells Lily Allen to 'f*** off' after she accuses him of profiting from Caroline Flack's death."

classifier = pipeline('sentiment-analysis', model='src/models/FN_Truth_Seeker_Model/checkpoint-6710')

print(classifier(text))