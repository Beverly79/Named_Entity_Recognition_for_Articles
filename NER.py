#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
# from seqeval.metrics import classification_report


# In[2]:


# Suppress warnings
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
import warnings
warnings.filterwarnings('ignore')


# ### Step 1: Load the Dataset

# https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus
# 
# Annotated Corpus for Named Entity Recognition using GMB(Groningen Meaning Bank) corpus for entity classification with enhanced and popular features by Natural Language Processing applied to the data set.

# In[3]:


try:
    datasets = load_dataset("rjac/kaggle-entity-annotated-corpus-ner-dataset")
    print("Dataset loaded successfully!")
    print(datasets)

    # Print the structure of the dataset
    print("Training set column names:", datasets["train"].column_names)
except Exception as e:
    print("Error loading dataset:", e)
    exit()


# ### Step 2: Define label names

# In[4]:


label_list = datasets["train"].features["ner_tags"].feature.names
print("Label List:", label_list)


# In[5]:


### Step 3: Load pre-trained BERT tokenizer


# In[6]:


# Tokenizer converts words into numerical format suitable for BERT
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# ### Step 4: Tokenize and align labels

# In[7]:


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens are ignored
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Tokens that are part of the same word are ignored
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization and label alignment to the dataset
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)


# ### Step 5: Split dataset into training and validation

# In[8]:


train_valid_split = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=1)
train_data = train_valid_split["train"]
val_data = train_valid_split["test"]


# ### Step 6: Load pre-trained BERT model for token classification

# In[9]:


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))


# ### Step 7: Define training arguments

# In[10]:


# Set the hyperparameters for fine-tuning the model
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    learning_rate=2e-5,  # Learning rate for optimizer
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=1,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    no_cuda=True,    #Train on CPU on a local machine
)


# ### Step 8: Data collator

# In[11]:


# Create a data collator to pad inputs and labels to the same length
data_collator = DataCollatorForTokenClassification(tokenizer)


# ### Step 9: Initialize Trainer

# In[12]:


# Use the Hugging Face Trainer class to handle training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# ### Step 10: Fine-tune the model

# In[ ]:


# Train the model using the Trainer
trainer.train()


# ### Step 11: Evaluate the model

# In[ ]:


predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = torch.argmax(torch.tensor(predictions), axis=2)

# Remove ignored index (-100) and map label IDs to names
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

# Print the classification report
print("Classification Report:")
print(classification_report(true_labels, true_predictions))

