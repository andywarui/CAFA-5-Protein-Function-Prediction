# %% [code] {"execution":{"iopub.status.busy":"2023-05-08T22:32:28.165267Z","iopub.execute_input":"2023-05-08T22:32:28.165806Z","iopub.status.idle":"2023-05-08T22:32:32.477240Z","shell.execute_reply.started":"2023-05-08T22:32:28.165764Z","shell.execute_reply":"2023-05-08T22:32:32.475969Z"}}
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

# Read in protein sequences
protein_sequences = []
for record in SeqIO.parse("/kaggle/input/cafa-5-protein-function-prediction/Train/train_sequences.fasta", "fasta"):
    protein_sequences.append(str(record.seq))

# Read in other files
train_annotations = pd.read_csv("/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv", sep="\t", header=None, names=["EntryID", "term", "aspect"])
train_taxonomy = pd.read_csv("/kaggle/input/cafa-5-protein-function-prediction/Train/train_taxonomy.tsv", sep="\t", header=None, names=["EntryID", "taxonomyID"])

# %% [code] {"execution":{"iopub.status.busy":"2023-05-08T22:32:35.035333Z","iopub.execute_input":"2023-05-08T22:32:35.035853Z","iopub.status.idle":"2023-05-08T22:37:33.801012Z","shell.execute_reply.started":"2023-05-08T22:32:35.035811Z","shell.execute_reply":"2023-05-08T22:37:33.799617Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2023-05-08T22:40:53.031306Z","iopub.execute_input":"2023-05-08T22:40:53.031904Z","iopub.status.idle":"2023-05-08T22:41:21.547433Z","shell.execute_reply.started":"2023-05-08T22:40:53.031859Z","shell.execute_reply":"2023-05-08T22:41:21.546289Z"}}
# Vectorize sequences and annotations
vocab = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'B': 20, 'Z': 21, 'X': 22, 'J': 23, 'O': 24, 'U': 25}
maxlen = 1000
train_sequences = np.zeros((len(protein_sequences), maxlen))
train_labels = np.zeros((len(protein_sequences), len(label_to_index)))
for i, sequence in enumerate(protein_sequences):
    for j, aa in enumerate(sequence):
        if j == maxlen:
            break
        train_sequences[i, j] = vocab[aa]
    annotations = seq_to_annotations.get(sequence, [])
    for annotation in annotations:
        train_labels[i, label_to_index[annotation]] = 1

# %% [code] {"execution":{"iopub.status.busy":"2023-05-08T22:43:08.440562Z","iopub.execute_input":"2023-05-08T22:43:08.441166Z","iopub.status.idle":"2023-05-08T22:43:11.438795Z","shell.execute_reply.started":"2023-05-08T22:43:08.441131Z","shell.execute_reply":"2023-05-08T22:43:11.437723Z"}}
# Split data into training and validation sets
train_sequences, val_sequences, train_labels, val_labels = train_test_split(train_sequences, train_labels, test_size=0.2)


# %% [code] {"execution":{"iopub.status.busy":"2023-05-08T22:43:15.093908Z","iopub.execute_input":"2023-05-08T22:43:15.094380Z","iopub.status.idle":"2023-05-08T22:43:15.766200Z","shell.execute_reply.started":"2023-05-08T22:43:15.094346Z","shell.execute_reply":"2023-05-08T22:43:15.764896Z"}}
# Define model architecture
vocab_size = 21
embedding_dim = 128
num_classes = len(label_to_index)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')
])

# %% [code] {"execution":{"iopub.status.busy":"2023-05-08T22:43:18.453509Z","iopub.execute_input":"2023-05-08T22:43:18.454700Z","iopub.status.idle":"2023-05-08T22:43:28.151992Z","shell.execute_reply.started":"2023-05-08T22:43:18.454640Z","shell.execute_reply":"2023-05-08T22:43:28.150323Z"}}
# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define hyperparameters
batch_size = 128
epochs = 20

# Train the model with early stopping
history = model.fit(train_sequences, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_sequences, val_labels), callbacks=[early_stopping])