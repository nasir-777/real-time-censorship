import cv2
import base64
import numpy as np
import time
import json
import threading
import queue
from confluent_kafka import Consumer, KafkaError
from collections import deque

# Kafka Consumer Configuration
conf = {
    'bootstrap.servers': 'pkc-l7pr2.ap-south-1.aws.confluent.cloud:9092',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': 'GAA5FXERGZHYVMM3',  # Replace with your actual API key
    'sasl.password': '2lNWNPSzQivRrJHYnTCN7TIWDi+39M5Jwm8pgkHqS2176OMEYFK3ZtZYrUlBc4+j',  # Replace with your actual API secret
    'group.id': 'video-frame-consumer',
    'enable.auto.commit': False,
    'auto.offset.reset': 'latest'
}

# Frame rate and delay settings
FRAME_RATE = 30
FRAME_DELAY = 1.0 / FRAME_RATE

# Initialize Kafka Consumer
consumer = Consumer(conf)
topic = 'barde2'
consumer.subscribe([topic])

# Buffer for frames and last sequence number
frame_buffer = deque(maxlen=10)  # Keep a buffer for a few frames
last_sequence = -1

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

frame_queue = queue.Queue(maxsize=10)
detection_history = deque(maxlen=10)

def detect_weapons(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    class_ids, confidences, boxes = [], [], []
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
                x, y = max(0, center_x - w // 2), max(0, center_y - h // 2)
                w, h = min(width - x, w), min(height - y, h)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        detection_history.append([(boxes[i], class_ids[i]) for i in indexes.flatten()])
    else:
        detection_history.append([])

    combined_boxes = []
    for frame_boxes in detection_history:
        combined_boxes.extend(frame_boxes)

    final_boxes = []
    for box, class_id in combined_boxes:
        if not any(np.allclose(box, b, atol=20) for b, _ in final_boxes):
            final_boxes.append((box, class_id))

    for (x, y, w, h), class_id in final_boxes:
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (15, 15), 30)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[class_id], 2)
        cv2.putText(frame, str(classes[class_id]), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, colors[class_id], 3)
    return frame

try:
    last_display_time = time.time()

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"Kafka error: {msg.error()}")
                continue

        try:
            # Decode the JSON message
            message = json.loads(msg.value().decode('utf-8'))
            print(f"Received message: {message}")  # Check received message structure
            timestamp = message.get('timestamp')
            frame_base64 = message.get('frame')
            sequence = message.get('sequence')

            if sequence is None:
                print("Error: 'sequence' not found in message.")
                continue  # Skip processing this message if 'sequence' is missing

            # Decode the base64-encoded image
            frame_data = base64.b64decode(frame_base64)
            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                if sequence > last_sequence:  # Only process if the sequence is greater
                    frame_buffer.append((timestamp, frame, sequence))
                    last_sequence = sequence

                # Sort buffer by sequence
                sorted_frames = sorted(frame_buffer, key=lambda x: x[2])

                # Display the frames in order
                current_time = time.time()
                for ts, frm, seq in sorted_frames:
                    if current_time - last_display_time >= FRAME_DELAY:
                        processed_frame = detect_weapons(frm)
                        cv2.imshow('Video Stream (Real-Time)', processed_frame)
                        last_display_time = current_time
                        # Remove the displayed frame from the buffer
                        frame_buffer.remove((ts, frm, seq))

                # Press 'q' to exit the video display
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting video stream.")
                    break
            else:
                print("Failed to decode frame.")
        
        except Exception as e:
            print(f"Error processing frame: {e}")

finally:
    consumer.close()
    cv2.destroyAllWindows()
