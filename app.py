import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Load the tokenizer and models
@st.cache_resource
def load_resources():
    from tensorflow.keras.preprocessing.text import Tokenizer

    # Load models
    RNN_model = load_model("RNN_lyrics_generator.h5")
    LSTM_model = load_model("LSTM_lyrics_generator.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer, RNN_model, LSTM_model

tokenizer, RNN_model, LSTM_model = load_resources()

# Constants
SEQ_LENGTH = 10
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Lyrics generation function
def generate_lyrics(model, seed_text, length=50):
    for _ in range(length):
        tokenized_input = tokenizer.texts_to_sequences([seed_text])[0]
        tokenized_input = pad_sequences([tokenized_input], maxlen=SEQ_LENGTH, padding="pre")
        predicted_idx = np.argmax(model.predict(tokenized_input, verbose=0))
        next_word = tokenizer.index_word.get(predicted_idx, "")
        seed_text += " " + next_word
    return seed_text

# Streamlit UI
st.title("🎤 Lyrics Generator")
st.markdown("Generate lyrics using a trained RNN or LSTM model!")

seed = st.text_input("Enter seed text:", value="Sometimes I think I'm a killer I scared you")
length = st.slider("Number of words to generate", min_value=10, max_value=100, value=50)
model_type = st.radio("Choose model", options=["RNN", "LSTM"])

if st.button("Generate Lyrics"):
    if model_type == "RNN":
        result = generate_lyrics(RNN_model, seed, length)
    else:
        result = generate_lyrics(LSTM_model, seed, length)
    st.subheader("🎶 Generated Lyrics:")
    st.write(result)