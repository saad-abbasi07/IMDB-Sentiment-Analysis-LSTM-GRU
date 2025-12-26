from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Request body structure
class Review(BaseModel):
    text: str

app = FastAPI()

# Enable CORS so browser requests work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is live"}

@app.post("/predict")
def predict_sentiment(review: Review):
    text = review.text

    # Dummy sentiment logic (replace with your trained model)
    if "love" in text.lower() or "great" in text.lower():
        sentiment = "Positive"
        score = 0.9
    else:
        sentiment = "Negative"
        score = 0.2

    return {"sentiment": sentiment, "score": score}
