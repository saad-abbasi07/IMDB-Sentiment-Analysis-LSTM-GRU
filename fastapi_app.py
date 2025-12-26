from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model & tokenizer
model = tf.keras.models.load_model("imdb_sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200

app = FastAPI(title="IMDB Sentiment API")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    seq = tokenizer.texts_to_sequences([review.text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    return {"sentiment": sentiment, "score": float(pred)}
