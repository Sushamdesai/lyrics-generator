import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
import os

# Download latest version of dataset
dataset_path = kagglehub.dataset_download("pratiksaha198/lyrics-generation")
print("Path to dataset files:", dataset_path)

# Print all files in path
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load dataset
data = pd.read_csv(dataset_path + "/LYRICS_DATASET.csv")
print("Artists in the data:\n", data['Artist Name'].value_counts())
print("Size of Dataset:", data.shape)

# Fill missing lyrics
data["Lyrics"] = data["Lyrics"].fillna("")

# Add text statistics
data['num_chars'] = data['Lyrics'].apply(len)
data['num_words'] = data['Lyrics'].apply(lambda x: len(x.split()))
data['num_sentences'] = data['Lyrics'].apply(lambda x: len(x.split('.')))
data.describe()

# Plot comparative song lengths
plt.figure(figsize=(15, 15))
sns.pairplot(data, hue="Artist Name", palette="plasma")

# Concatenate all lyrics into one text corpus
Corpus = " ".join(data.Lyrics).lower()
print("Number of unique characters:", len(set(Corpus)))
print("The unique characters:", sorted(set(Corpus)))

# Function to clean lyrics
def clean_lyrics(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keeps only letters, numbers, and spaces

# Apply cleaning function
data["Lyrics"] = data["Lyrics"].apply(clean_lyrics)

# Tokenize at word level
tokenizer = Tokenizer()
tokenizer.fit_on_texts([Corpus])

# Convert text to integer sequences
word_sequences = tokenizer.texts_to_sequences([Corpus])[0]

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

# Define sequence length
seq_length = 10  # Training on 10-word sequences

# Create input-output pairs
sequences = []
next_words = []
for i in range(len(word_sequences) - seq_length):
    sequences.append(word_sequences[i:i + seq_length])
    next_words.append(word_sequences[i + seq_length])

X = np.array(sequences)
y = np.array(next_words)
y = to_categorical(y, num_classes=vocab_size)  # One-hot encoding

# Tokenize at word level
tokenizer = Tokenizer()
tokenizer.fit_on_texts([Corpus])

# Save the tokenizer
import json
tokenizer_json = json.dumps(tokenizer.word_index)
with open("tokenizer.json", "w") as f:
    f.write(tokenizer_json)
print("Tokenizer saved as tokenizer.json")


# ==== Check if LSTM model exists before training ====
if os.path.exists("LSTM_lyrics_generator.h5"):
    print("Loading saved LSTM model...")
    LSTM_model = load_model("LSTM_lyrics_generator.h5",compile=False)
else:
    print("Training LSTM model...")
    LSTM_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_length),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(256),
        Dense(vocab_size, activation="softmax")  # Predict next word
    ])
    LSTM_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    LSTM_model.fit(X, y, epochs=30, batch_size=128)
    LSTM_model.save("LSTM_lyrics_generator.h5")  # Save the trained model

# ==== Check if RNN model exists before training ====
if os.path.exists("RNN_lyrics_generator.h5"):
    print("Loading saved RNN model...")
    RNN_model = load_model("RNN_lyrics_generator.h5",safe_mode = False)
else:
    print("Training RNN model...")
    RNN_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_length),
        SimpleRNN(256, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        SimpleRNN(256, activation="tanh"),
        Dense(vocab_size, activation="softmax")
    ])
    RNN_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    RNN_model.fit(X, y, epochs=30, batch_size=128)
    RNN_model.save("RNN_lyrics_generator.h5")  # Save the trained model

# Function to generate lyrics using RNN
def generate_lyrics(seed_text, length=100):
    RNN_model = load_model('RNN_lyrics_generator.h5', safe_mode=False)
    RNN_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    for _ in range(length):
        tokenized_input = tokenizer.texts_to_sequences([seed_text])[0]
        tokenized_input = pad_sequences([tokenized_input], maxlen=seq_length, padding="pre")
        predicted_idx = np.argmax(RNN_model.predict(tokenized_input, verbose=0))
        next_word = tokenizer.index_word.get(predicted_idx, "")
        seed_text += " " + next_word
    return seed_text

print(generate_lyrics("Sometimes I think I'm a killer I scared you the way you looked at me I never wanted to hurt you But now I see the pain in your eyes And I can't take it back...The echoes of silence haunt my mind Shadows creeping, love left behind Your tears fall like the pouring rainBut my hands are stained with pain", length=100))

# Function to generate lyrics using LSTM
def generate_lyrics_with_LSTM(seed_text, length=100):
    LSTM_model = load_model('LSTM_lyrics_generator.h5')
    LSTM_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    for _ in range(length):
        tokenized_input = tokenizer.texts_to_sequences([seed_text])[0]
        tokenized_input = pad_sequences([tokenized_input], maxlen=seq_length, padding="pre")
        predicted_idx = np.argmax(LSTM_model.predict(tokenized_input, verbose=0))
        next_word = tokenizer.index_word.get(predicted_idx, "")
        seed_text += " " + next_word
    return seed_text

print(generate_lyrics_with_LSTM("Sometimes I think I'm a killer I scared you the way you looked at me I never wanted to hurt you But now I see the pain in your eyes And I can't take it back...The echoes of silence haunt my mind Shadows creeping, love left behind Your tears fall like the pouring rainBut my hands are stained with pain", length=100))
