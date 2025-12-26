from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load model and tokenizer
model = tf.keras.models.load_model("imdb_sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200  # same length used during training

app = FastAPI()

class Review(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API is live"}

@app.post("/predict")
def predict(review: Review):
    # Preprocess input text
    seq = tokenizer.texts_to_sequences([review.text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Make prediction
    score = float(model.predict(padded)[0][0])
    sentiment = "Positive" if score >= 0.5 else "Negative"
    
    return {"sentiment": sentiment, "score": score}
