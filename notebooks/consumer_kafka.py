import os
from kafka import KafkaConsumer

# Set up Kafka Consumer
consumer = KafkaConsumer(
    'file-topic',  # Same topic used by the producer
    bootstrap_servers='localhost:9092',  # Update with your Kafka server
    auto_offset_reset='earliest',  # Start from the earliest message
    enable_auto_commit=True,
    group_id='file-group'
)

destination_folder = '../data_written'

def receive_and_write_files():
    for message in consumer:
        filename = message.key.decode()
        file_content = message.value
        
        # Write the file content to the destination folder
        dest_file_path = os.path.join(destination_folder, filename)
        
        with open(dest_file_path, 'wb') as file:
            file.write(file_content)
        
        print(f"Received and wrote file: {filename}")

receive_and_write_files()
