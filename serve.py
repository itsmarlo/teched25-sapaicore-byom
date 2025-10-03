"""
Flask-based Inference Server for SAP AI Core
Serves the trained Hugging Face model
"""
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
label_map = None

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer, label_map
    
    model_path = os.getenv('MODEL_PATH', '/app/model')
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # Load label mapping
        label_map_path = os.path.join(model_path, 'label_map.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
        else:
            # Default label map
            label_map = {
                "0": "Price Variance",
                "1": "Quantity Mismatch",
                "2": "Missing PO Reference",
                "3": "Three-Way Match Failure",
                "4": "Supplier Issues"
            }
        
        logger.info("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200

@app.route('/v1/models', methods=['GET'])
def models():
    """List available models"""
    return jsonify({
        "models": [{
            "name": "invoice-classifier",
            "version": "1.0.0",
            "status": "ready" if model is not None else "loading"
        }]
    }), 200

@app.route('/v2/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint compatible with SAP AI Core
    
    Expected input format:
    {
        "invoice_text": "Invoice amount exceeds PO by 15%"
    }
    
    or batch prediction:
    {
        "invoices": [
            {"text": "Invoice 1 description"},
            {"text": "Invoice 2 description"}
        ]
    }
    """
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded"
            }), 503
        
        data = request.get_json()
        
        # Handle single prediction
        if 'invoice_text' in data:
            text = data['invoice_text']
            results = [predict_single(text)]
        
        # Handle batch prediction
        elif 'invoices' in data:
            invoices = data['invoices']
            results = [predict_single(inv.get('text', '')) for inv in invoices]
        
        else:
            return jsonify({
                "error": "Invalid input format. Expected 'invoice_text' or 'invoices'"
            }), 400
        
        return jsonify({
            "predictions": results,
            "model": "invoice-classifier",
            "version": "1.0.0"
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

def predict_single(text):
    """Predict blocking reason for a single invoice"""
    
    # Tokenize input
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
    # Get all probabilities
    all_probs = {
        label_map.get(str(i), f"Class_{i}"): float(probs[0][i])
        for i in range(len(probs[0]))
    }
    
    return {
        "text": text,
        "predicted_class": predicted_class,
        "predicted_label": label_map.get(str(predicted_class), f"Class_{predicted_class}"),
        "confidence": float(confidence),
        "all_probabilities": all_probs
    }

@app.route('/v2/explain', methods=['POST'])
def explain():
    """
    Explanation endpoint for model predictions
    Provides reasoning for classification
    """
    try:
        data = request.get_json()
        text = data.get('invoice_text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        prediction = predict_single(text)
        
        # Add explanation
        explanation = generate_explanation(
            text, 
            prediction['predicted_label'],
            prediction['confidence']
        )
        
        prediction['explanation'] = explanation
        
        return jsonify(prediction), 200
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_explanation(text, predicted_label, confidence):
    """Generate human-readable explanation"""
    
    keywords = {
        "Price Variance": ["price", "amount", "exceeds", "variance", "EUR", "USD"],
        "Quantity Mismatch": ["quantity", "units", "items", "received", "mismatch"],
        "Missing PO Reference": ["PO", "purchase order", "reference", "missing", "not found"],
        "Three-Way Match Failure": ["goods receipt", "three-way", "match", "GR", "posted"],
        "Supplier Issues": ["supplier", "vendor", "blocked", "credit hold", "compliance"]
    }
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in keywords.get(predicted_label, []):
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    explanation = {
        "reasoning": f"Classified as '{predicted_label}' with {confidence*100:.1f}% confidence",
        "detected_keywords": found_keywords,
        "recommendation": get_recommendation(predicted_label)
    }
    
    return explanation

def get_recommendation(label):
    """Provide action recommendations based on classification"""
    
    recommendations = {
        "Price Variance": "Review purchase order pricing and obtain approval for variance if needed",
        "Quantity Mismatch": "Verify goods receipt quantities and contact supplier if discrepancy exists",
        "Missing PO Reference": "Request valid PO number from supplier and update invoice",
        "Three-Way Match Failure": "Ensure goods receipt is posted before processing invoice",
        "Supplier Issues": "Check vendor master data and resolve any blocking issues"
    }
    
    return recommendations.get(label, "Review invoice details and take appropriate action")

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model. Exiting...")
        exit(1)
    
    # Start server
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)