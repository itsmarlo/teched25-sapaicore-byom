import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(MODEL_PATH, "config.json")) as f:
    cfg = json.load(f)
id2label = cfg.get("id2label") or {i: str(i) for i in range(cfg.get("num_labels", 5))}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

app = FastAPI()

class PredictPayload(BaseModel):
    texts: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictPayload):
    enc = tokenizer(payload.texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        preds = logits.argmax(dim=-1).cpu().tolist()

    results = []
    for i, p in enumerate(preds):
        results.append({
            "text": payload.texts[i],
            "label_id": int(p),
            "label": id2label.get(str(p), id2label.get(p, str(p))),
            "probs": probs[i]
        })
    return {"results": results}
