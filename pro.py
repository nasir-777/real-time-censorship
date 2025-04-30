#title abstrac breef methedology result
#ver 1.2
import cv2
import base64
import time
from confluent_kafka import Producer
import json

# Kafka Producer Configuration
conf = {
    'bootstrap.servers': 'pkc-l7pr2.ap-south-1.aws.confluent.cloud:9092',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': 'GAA5FXERGZHYVMM3',  # Replace with your actual API key
    'sasl.password': '2lNWNPSzQivRrJHYnTCN7TIWDi+39M5Jwm8pgkHqS2176OMEYFK3ZtZYrUlBc4+j',  # Replace with your actual API secret
    'client.id': 'video-frame-producer'
}
producer = Producer(conf)
topic = 'barde2'

def acked(err, msg):
    if err is not None:
        print(f"Failed to deliver message: {err}")
    else:
        print(f"Message produced: {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}")

# Open the webcam
webCam = cv2.VideoCapture(0)

if not webCam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Measure FPS from the webcam
fps = webCam.get(cv2.CAP_PROP_FPS)
if fps == 0:  # Some cameras don't provide FPS via OpenCV
    fps = 30  # Default to 30 FPS if not provided
frame_interval = 1.0 / fps

frame_count = 0
start_time = time.time()

try:
    while True:
        success, frame = webCam.read()
        
        if not success:
            print("Failed to capture frame.")
            break

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            # Convert the image to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Attach a timestamp and sequence number to the frame
            message = {
                'timestamp': time.time(),  # Current time in seconds
                'frame': frame_base64,
                'sequence': frame_count  # Add sequence number
            }

            # Print the message for debugging
            print(f"Producing message: {message}")  # Check the message structure

            # Produce message to Kafka
            producer.produce(topic, json.dumps(message), callback=acked)
            producer.poll(0)

            frame_count += 1
            print(f"Produced frame {frame_count}")

            # Display the video output
            cv2.imshow("Output", frame)

        # Throttle to match the frame rate of the camera
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)
        start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    webCam.release()
    cv2.destroyAllWindows()
    producer.flush()