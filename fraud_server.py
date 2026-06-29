# src/fraud_server.py
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from kafka import KafkaConsumer
import joblib
import pandas as pd
import sys

# Append root directory to access global configurations safely
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, FRAUD_TOPIC, SERVER_HOST, SERVER_PORT

app = FastAPI(title="Standalone Fraud Detection System")

# Set up templates directory path maps
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Load our pre-trained AI classification brain
model_path = os.path.join(base_dir, "data", "fraud_model.pkl")
if os.path.exists(model_path):
    print(f"🧠 Loading Machine Learning Classifier Core from: {model_path}")
    fraud_brain = joblib.load(model_path)
else:
    print(f"⚠️ WARNING: Trained model file missing at {model_path}. Please run train_model.py first!")
    fraud_brain = None

# Look for this block around line 31 in src/fraud_server.py and update it:

@app.get("/", response_class=HTMLResponse)
async def serve_security_dashboard(request: Request):
    """Serves the central security monitoring command dashboard UI"""
    # Updated syntax to match new FastAPI/Jinja2 standard specifications:
    return templates.TemplateResponse(request, name="fraud_dashboard.html")

@app.websocket("/ws/fraud-stream")
async def process_fraud_stream(websocket: WebSocket):
    """Manages full-duplex real-time inference and client dashboard broadcasting"""
    await websocket.accept()
    print("🔌 Security Monitoring Dashboard connected to real-time engine.")

    # Initialize the specific independent Kafka consumer pipeline
    try:
        consumer = KafkaConsumer(
            FRAUD_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
    except Exception as e:
        print(f"❌ Consumer unable to link to Kafka network infrastructure: {e}")
        await websocket.close()
        return

    try:
        while True:
            # Non-blocking poll request sequence
            msg_pack = consumer.poll(timeout_ms=100)
            
            for tp, messages in msg_pack.items():
                for message in messages:
                    raw_data = message.value
                    
                    # 1. Isolate structural columns required for our model feature matrix
                    input_features = {k: v for k, v in raw_data.items() if k not in ['Time', 'Class', 'account_id']}
                    
                    # Convert to a single-row DataFrame format for scikit-learn
                    feature_dataframe = pd.DataFrame([input_features])
                    
                    # 2. Fire instant calculation logic through our predictive model
                    prediction = 0
                    confidence_score = 1.0
                    
                    if fraud_brain is not None:
                        # Return binary value (0 = Safe transaction, 1 = FRAUD)
                        prediction = int(fraud_brain.predict(feature_dataframe)[0])
                        
                        # Extract prediction probability values
                        probabilities = fraud_brain.predict_proba(feature_dataframe)[0]
                        confidence_score = float(probabilities[prediction])

                    # 3. Build a structured analytics broadcast packet
                    analytics_payload = {
                        "account_id": raw_data["account_id"],
                        "amount": round(raw_data["Amount"], 2),
                        "is_fraud": prediction,
                        "confidence": round(confidence_score * 100, 2)
                    }
                    
                    # 4. Transmit data packet over the active open WebSocket pipe
                    await websocket.send_json(analytics_payload)
            
            # Relinquish thread timing to keep the server running smoothly
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("🔌 Client disconnected from security channel pipeline.")
    finally:
        consumer.close()

# Optimized runner sequence to support direct "python src/fraud_server.py" calls
if __name__ == "__main__":
    import uvicorn
    
    print("⏳ Starting FastAPI Fraud Detection Server...")
    print(f"🌐 Access your dashboard at: http://{SERVER_HOST}:{SERVER_PORT}")
    
    # Passing the exact package-style string format so uvicorn reloader functions smoothly
    uvicorn.run("src.fraud_server:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)