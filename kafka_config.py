# config/kafka_config.py

# Kafka Cluster Configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
FRAUD_TOPIC = 'banking_transaction_stream'

# FastAPI Server Routing Configuration
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8050  # Changed to 8050 so it never clashes with other local web apps