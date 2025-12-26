from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model("imdb_sentiment_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

max_len = 200

app = FastAPI()

class Review(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API is live"}

@app.post("/predict")
def predict(review: Review):
    seq = tokenizer.texts_to_sequences([review.text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    score = model.predict(padded)[0][0]
    sentiment = "Positive" if score > 0.5 else "Negative"
    return {"sentiment": sentiment, "score": float(score)}
