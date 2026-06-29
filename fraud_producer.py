# src/fraud_producer.py
import os
import time
import json
import pandas as pd
from kafka import KafkaProducer
import sys

# Append root directory to access global configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, FRAUD_TOPIC

def start_transaction_stream():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")

    print("📡 Initializing Bank Terminal Transaction Network...")
    if not os.path.exists(data_path):
        print(f"❌ Error: Cannot find dataset at {data_path}")
        return

    # Initialize Kafka Producer client with JSON serialization
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"❌ Kafka Connection Failure. Ensure Docker containers are running. Details: {e}")
        return

    # Read the dataset rows
    df = pd.read_csv(data_path)
    print(f" Loaded {len(df)} simulation records. Commencing active streaming...")

    # Infinite stream loop: Shuffle data rows to simulate random live swiping terminals
    while True:
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        for _, row in shuffled_df.iterrows():
            # Convert row matrix into a standard serializable dictionary
            transaction_packet = row.to_dict()
            
            # Extract basic identifiers for visualization logging
            # Assigning a mock account ID since the real dataset features are anonymized V1-V28
            account_id = int(100000 + (row.name % 899999))
            transaction_packet["account_id"] = account_id

            # Broadcast onto the Kafka infrastructure
            producer.send(FRAUD_TOPIC, value=transaction_packet)
            
            print(f" [Terminal Broadcast] Account #{account_id} | Amount: ${transaction_packet['Amount']} -> Transmitted")
            
            # Rest for 400 milliseconds to simulate readable real-time velocity
            time.sleep(0.4)

if __name__ == "__main__":
    start_transaction_stream()