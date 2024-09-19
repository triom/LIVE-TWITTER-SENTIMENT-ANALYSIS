# import os
# from kafka import KafkaConsumer

# # Set up Kafka Consumer
# consumer = KafkaConsumer(
#     'file-topic',  # Same topic used by the producer
#     bootstrap_servers='localhost:9092',  # Update with your Kafka server
#     fetch_max_bytes=52428800,  # 50 MB (increase fetch size for larger batches)
#     max_poll_records=500,
#     auto_offset_reset='earliest',  # Start from the earliest message
#     enable_auto_commit=True,
#     group_id='file-group'
# )


# destination_folder = '../data_written'

# def receive_and_write_files():
#     for message in consumer:
#         filename = message.key.decode()
#         file_content = message.value
        
#         # Write the file content to the destination folder
#         dest_file_path = os.path.join(destination_folder, filename)
        
#         with open(dest_file_path, 'wb') as file:
#             file.write(file_content)
        
#         print(f"Received and wrote file: {filename}")

# receive_and_write_files()



# **********************************


# import os
# from kafka import KafkaConsumer

# # Kafka Consumer configuration
# consumer = KafkaConsumer(
#     'file-topic',  # Topic name
#     bootstrap_servers='localhost:9092',  # Update with your Kafka server
#     group_id='file-consumer-group',  # Consumer group ID
#     auto_offset_reset='earliest',  # Start reading at the earliest offset
#     enable_auto_commit=True,  # Enable automatic offset commit
# )

# destination_folder = './received_files'  # Destination folder to save files

# # Ensure the destination folder exists
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# def save_file(key, content):
#     # Convert the key from bytes to string (filename)
#     filename = key.decode('utf-8')
#     file_path = os.path.join(destination_folder, filename)
    
#     # Save the file content
#     with open(file_path, 'wb') as file:
#         file.write(content)
    
#     print(f"Saved file: {filename} to {destination_folder}")

# # Listen for messages from Kafka topic and save them as files
# try:
#     for message in consumer:
#         # Retrieve the file key (filename) and value (file content)
#         key = message.key  # This is the filename
#         content = message.value  # This is the file content
        
#         save_file(key, content)

# except Exception as e:
#     print(f"Error occurred: {e}")

# finally:
#     # Close the consumer connection when done
#     consumer.close()


from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'file-topic',  # Ensure this is the correct topic
    bootstrap_servers=['localhost:9092'],  # Adjust to your Kafka broker
    auto_offset_reset='earliest',  # Start from the earliest message
    group_id='your_consumer_group'
)

# Consume messages from the topic
for message in consumer:
    print(f"Received message: {message.value.decode('utf-8')}")
