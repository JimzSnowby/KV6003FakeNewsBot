Author: James Sowerby
Student ID: W21023500

Virtual Python environment must be setup first! (venv)
Install packages with !pip install.

Packages:
transformers
datasets
evaluate
accelerate
pipeline
pandas
numpy
scikit-learn
tweepy

Training a new model:
1) load a dataset and split with train_test_split function in model.py.
2) load tokenizer with AutoTokenizer.from_pretrained() and choose a model from https://huggingface.co/models?pipeline_tag=text-classification&sort=trending.
3) load the model with AutoModelForSequenceClassification.from_pretrained using the same model from above.
4) set output directory.
5) run.
    Note) per_device_train_batch_size and per_device_eval_batch_size may need to be adjusted depending on the GPU VRAM.

Making predictions:
In main.py, change the "classifier" variable to the location of the desired model and its checkpoint, then run the file.
    e.g model='src/models/FN_TS_MobileBERT/checkpoint-26840'

If using a paid developer account, use the appropriate function.
