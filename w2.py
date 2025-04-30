# import cv2
# import numpy as np


# # Load Yolo
# # Download weight file(yolov3_training_2000.weights) from this link :- https://drive.google.com/file/d/10uJEsUpQI3EmD98iwrwzbD4e19Ps-LHZ/view?usp=sharing
# net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
# classes = ["Weapon"]
# # with open("coco.names", "r") as f:
# #     classes = [line.strip() for line in f.readlines()]

# # layer_names = net.getLayerNames()
# # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# output_layer_names = net.getUnconnectedOutLayersNames()
# colors = np.random.uniform(0, 255, size=(len(classes), 3))


# # Loading image
# # img = cv2.imread("room_ser.jpg")
# # img = cv2.resize(img, None, fx=0.4, fy=0.4)

# # Enter file name for example "ak47.mp4" or press "Enter" to start webcam
# def value():
#     val = input("Enter file name or press enter to start webcam : \n")
#     if val == "":
#         val = 0
#     return val


# # for video capture
# cap = cv2.VideoCapture(value())

# # val = cv2.VideoCapture()
# while True:
#     _, img = cap.read()
#     if not _:
#         print("Error: Failed to read a frame from the video source.")
#         break
#     height, width, channels = img.shape
#     # width = 512
#     # height = 512

#     # Detecting objects
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#     net.setInput(blob)
#     outs = net.forward(output_layer_names)

#     # Showing information on the screen
#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     print(indexes)
#     if indexes == 0: print("weapon detected in frame")
#     font = cv2.FONT_HERSHEY_PLAIN
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             color = colors[class_ids[i]]
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

#     # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()



# ------------------------------------------------------------------------------------------------------------

# import cv2
# import numpy as np
# from collections import deque

# # Load Yolo
# net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
# classes = ["Weapon"]
# output_layer_names = net.getUnconnectedOutLayersNames()
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

# # Maintain a history of bounding boxes for smoothing
# detection_history = deque(maxlen=10)  # Stores bounding boxes for the last 10 frames

# # Function to get input video or webcam
# def value():
#     val = input("Enter file name or press enter to start webcam : \n")
#     if val == "":
#         val = 0
#     return val

# # Video capture
# cap = cv2.VideoCapture(value())

# while True:
#     ret, img = cap.read()
#     if not ret:
#         print("Error: Failed to read a frame from the video source.")
#         break

#     height, width, channels = img.shape

#     # Detecting objects
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layer_names)

#     # Initialize lists for detection results
#     class_ids = []
#     confidences = []
#     boxes = []

#     # Process detections
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 # Ensure coordinates are within image boundaries
#                 x = max(0, x)
#                 y = max(0, y)
#                 w = min(width - x, w)
#                 h = min(height - y, h)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply Non-Maximum Suppression
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Update detection history
#     if len(indexes) > 0:
#         detection_history.append([(boxes[i], class_ids[i]) for i in indexes.flatten()])
#     else:
#         detection_history.append([])  # No detection in the current frame

#     # Combine detections from history for smoothing
#     combined_boxes = []
#     for frame_boxes in detection_history:
#         combined_boxes.extend(frame_boxes)

#     # Remove duplicates and average box positions over history
#     final_boxes = []
#     for box, class_id in combined_boxes:
#         if not any(np.allclose(box, b, atol=20) for b, _ in final_boxes):  # Remove near-duplicates
#             final_boxes.append((box, class_id))

#     font = cv2.FONT_HERSHEY_PLAIN

#     # Iterate through final smoothed boxes
#     for (x, y, w, h), class_id in final_boxes:
#         label = str(classes[class_id])
#         color = colors[class_id]

#         # Apply blur to the bounding box region
#         roi = img[y:y+h, x:x+w]

#         if roi.size > 0:  # Check if the ROI is valid
#             blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)  # Adjust kernel size for more/less blur
#             img[y:y+h, x:x+w] = blurred_roi  # Replace ROI with blurred ROI

#         # Optionally, draw a rectangle around the blurred region
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

#     # Display the frame
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == 27:  # Press 'Esc' to exit
#         break

# cap.release()
# cv2.destroyAllWindows()


# works 100%

# ---------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import threading
import queue
from collections import deque

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set up VideoCapture
cap = cv2.VideoCapture(input("Enter file name or press enter to start webcam: ") or 0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

frame_queue = queue.Queue(maxsize=10)
detection_history = deque(maxlen=10)

def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

frame_counter = 0
while True:
    if frame_queue.empty():
        continue
    img = frame_queue.get()

    frame_counter += 1
    if frame_counter % 2 != 0:  # Skip every other frame
        continue

    img = cv2.resize(img, (640, 360))  # Resize for faster processing
    height, width, _ = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
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
        roi = img[y:y+h, x:x+w]
        if roi.size > 0:
            img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (15, 15), 30)
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_id], 2)
        cv2.putText(img, str(classes[class_id]), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, colors[class_id], 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

