# import os
# from kafka import KafkaProducer

# # Set up Kafka Producer
# producer = KafkaProducer(bootstrap_servers='localhost:9092')  # Update with your Kafka server

# source_folder = '../data'

# def read_and_send_files(folder):
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)
#         if os.path.isfile(file_path):
#             # Read the file contents
#             with open(file_path, 'rb') as file:
#                 file_content = file.read()
            
#             # Send file content to Kafka topic
#             producer.send('file-topic', value=file_content, key=filename.encode())  # Send the filename as the key
            
#             print(f"Sent file: {filename} to Kafka topic")

# read_and_send_files(source_folder)

from kafka import KafkaProducer
import csv

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])  # Adjust to your broker

# Read CSV file and produce messages
with open('final_df.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        message = ','.join(row)  # Combine CSV row into a string
        producer.send('your_topic', value=message.encode('utf-8'))  # Send to Kafka topic

producer.flush()  # Ensure all messages are sent before exiting
producer.close()
