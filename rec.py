import cv2
import base64
import numpy as np
import time
import re
import easyocr
import pandas as pd
from confluent_kafka import Consumer, KafkaError
import json
import threading
import queue

# Kafka Consumer configuration
conf = {
    'bootstrap.servers': 'pkc-l7pr2.ap-south-1.aws.confluent.cloud:9092',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': 'GAA5FXERGZHYVMM3',
    'sasl.password': '2lNWNPSzQivRrJHYnTCN7TIWDi+39M5Jwm8pgkHqS2176OMEYFK3ZtZYrUlBc4+j',
    'group.id': 'video-frame-consumer',
    'enable.auto.commit': False,
    'auto.offset.reset': 'latest'
}

# Frame rate and delay settings
FRAME_RATE = 30
FRAME_DELAY = 1.0 / FRAME_RATE

# Sensitive patterns (credit card numbers)
sensitive_patterns = [r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b']

# Function to load malicious words from Hindi.csv
def load_hindi_words(csv_file):
    df = pd.read_csv(csv_file, header=None)
    return df[0].tolist()

# Function to load English profanity words from a CSV
def load_english_words(csv_file):
    df = pd.read_csv(csv_file, header=None)
    return df[0].tolist()

# Initialize OCR Reader (optional: use_cuda=False to ensure smoother CPU-based detection)
def initialize_ocr(use_cuda=True):
    return easyocr.Reader(['en', 'hi'], gpu=use_cuda)

# Initialize OCR and load malicious words
use_cuda = True  # Adjust to False if CUDA is causing delay
reader = initialize_ocr(use_cuda)
malicious_words = ["STARTUP", "PLAN", "Kind", "Hello", "bye"] + load_hindi_words('Hindi.csv') + load_english_words('profanity_en.csv')

# Initialize Kafka Consumer
consumer = Consumer(conf)
consumer.subscribe(['barde2'])

# Buffer for frames and last sequence number
frame_queue = queue.Queue(maxsize=10)  # Use a Queue with limited size to control frame flow
last_sequence = -1

# Persistent blur storage (keeps track of blurred regions)
persistent_blur = {}

# Function to blur regions of the image
def blur_region(image, top_left, bottom_right):
    sub_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    blurred_sub_img = cv2.GaussianBlur(sub_img, (51, 51), 0)
    image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_sub_img

# Function to process each frame and track moving text for continuous blurring
def process_frame(frame, result):
    updated_blur_regions = []

    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        blur_text = False

        for pattern in sensitive_patterns:
            if re.search(pattern, text):
                blur_text = True
                updated_blur_regions.append((top_left, bottom_right))
                break

        for word in malicious_words:
            if word.upper() in text.upper():
                blur_text = True
                updated_blur_regions.append((top_left, bottom_right))
                break

        if not blur_text:
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Apply blur for all regions in updated_blur_regions
    for top_left, bottom_right in updated_blur_regions:
        blur_region(frame, top_left, bottom_right)

# Function to handle frames in queue and apply OCR
def handle_frame_queue():
    while True:
        if not frame_queue.empty():
            timestamp, frame = frame_queue.get()
            result = reader.readtext(frame)  # OCR on the frame
            process_frame(frame, result)

            cv2.imshow('Real-Time Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_queue.task_done()

# Thread to handle frame processing
frame_handler_thread = threading.Thread(target=handle_frame_queue)
frame_handler_thread.daemon = True
frame_handler_thread.start()

# Main Kafka and video processing loop
try:
    while True:
        msg = consumer.poll(timeout=0.2)  # Poll Kafka messages
        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"Kafka error: {msg.error()}")
                continue

        try:
            message = json.loads(msg.value().decode('utf-8'))
            frame_base64 = message.get('frame')
            sequence = message.get('sequence')

            if sequence is None:
                print("Error: 'sequence' not found in message.")
                continue

            # Decode frame from base64
            frame_data = base64.b64decode(frame_base64)
            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None and sequence > last_sequence:
                last_sequence = sequence

                if frame_queue.full():
                    # Drop frames if queue is full to reduce lag
                    print("Frame dropped to avoid lag.")
                else:
                    # Add frame to the queue for processing
                    frame_queue.put((message.get('timestamp'), frame))

        except Exception as e:
            print(f"Error processing frame: {e}")

finally:
    consumer.close()
    cv2.destroyAllWindows()
