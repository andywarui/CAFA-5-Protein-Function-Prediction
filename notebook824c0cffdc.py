


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from Bio import SeqIO

# Read in protein sequences
protein_sequences = []
for record in SeqIO.parse("/kaggle/input/cafa-5-protein-function-prediction/Train/train_sequences.fasta", "fasta"):
    protein_sequences.append(str(record.seq))

# Read in other files
train_annotations = pd.read_csv("/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv", sep="\t", header=None, names=["EntryID", "term", "aspect"])
train_taxonomy = pd.read_csv("/kaggle/input/cafa-5-protein-function-prediction/Train/train_taxonomy.tsv", sep="\t", header=None, names=["EntryID", "taxonomyID"])

# Merge the annotations and taxonomy files
train_data = pd.merge(train_annotations, train_taxonomy, on="EntryID")

# Create dictionary mapping sequences to annotations
seq_to_annotations = {}
for index, row in train_data.iterrows():
    sequence = row["EntryID"]
    annotations = row["term"].split("; ")
    seq_to_annotations[sequence] = annotations

# Create dictionary mapping annotations to indices
all_annotations = set()
for annotations in seq_to_annotations.values():
    all_annotations.update(annotations)
label_to_index = {label: i for i, label in enumerate(sorted(all_annotations))}

# Vectorize sequences and annotations
vocab = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'B': 20, 'Z': 21, 'X': 22, 'J': 23, 'O': 24, 'U': 25}
maxlen = 1000
train_sequences = np.zeros((len(protein_sequences), maxlen))
train_labels = np.zeros((len(protein_sequences), len(label_to_index)))
for i, sequence in enumerate(protein_sequences):
    for j, aa in enumerate(sequence):
        if j == maxlen:
            break
        if aa in vocab:
            train_sequences[i, j] = vocab[aa]
    annotations = seq_to_annotations.get(sequence, [])
    for annotation in annotations:
        if annotation in label_to_index:
            train_labels[i, label_to_index[annotation]] = 1

# Split data into training and validation sets
train_sequences, val_sequences, train_labels, val_labels = train_test_split(train_sequences, train_labels, test_size=0.2)

# Define the model
inputs = Input(shape=(maxlen,))
x = Embedding(len(vocab), 50)(inputs)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(
