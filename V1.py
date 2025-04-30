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
import torch

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

# Sensitive patterns (e.g., credit card numbers)
sensitive_patterns = [r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b']

# Load malicious words (example)
malicious_words = ["STARTUP", "PLAN", "Hello", "bye"]

# Initialize OCR Reader
reader = easyocr.Reader(['en', 'hi'], gpu=True)

# Initialize YOLO for weapon detection
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Frame queue
frame_queue = queue.Queue(maxsize=10)
last_sequence = -1

# Kafka consumer
consumer = Consumer(conf)
consumer.subscribe(['barde'])

# Function to blur sensitive regions
def blur_region(image, top_left, bottom_right):
    sub_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    blurred_sub_img = cv2.GaussianBlur(sub_img, (51, 51), 0)
    image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_sub_img

# Function to process frames for sensitive text
def process_sensitive_text(frame, result):
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        blur_text = any(re.search(pattern, text) for pattern in sensitive_patterns) or \
                    any(word.upper() in text.upper() for word in malicious_words)

        if blur_text:
            blur_region(frame, top_left, bottom_right)
        else:
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Function to detect weapons using YOLO
def detect_weapons(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

# Function to process frames
def process_frame():
    while True:
        if not frame_queue.empty():
            timestamp, frame = frame_queue.get()
            result = reader.readtext(frame)
            process_sensitive_text(frame, result)
            detect_weapons(frame)
            cv2.imshow('Processed Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_queue.task_done()

# Start frame processing thread
thread = threading.Thread(target=process_frame)
thread.daemon = True
thread.start()

# Main Kafka loop
try:
    while True:
        msg = consumer.poll(timeout=0.2)
        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"Kafka error: {msg.error()}")
                continue

        message = json.loads(msg.value().decode('utf-8'))
        frame_base64 = message.get('frame')
        sequence = message.get('sequence')

        if sequence is None or frame_base64 is None:
            print("Error: Missing 'sequence' or 'frame'.")
            continue

        if sequence > last_sequence:
            last_sequence = sequence
            frame_data = base64.b64decode(frame_base64)
            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                if frame_queue.full():
                    print("Frame dropped to avoid lag.")
                else:
                    frame_queue.put((message.get('timestamp'), frame))
finally:
    consumer.close()
    cv2.destroyAllWindows()
