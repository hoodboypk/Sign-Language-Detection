from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os
import subprocess
from gtts import gTTS

model = YOLO(r"E:\ML Projects\Sign Language Detection\Sign Language.v1i.yolov8\runs\detect\train2\weights\best.pt")

# Replace this with the correct path to Windows Media Player
wmplayer_path = r"C:\Program Files (x86)\Windows Media Player\wmplayer.exe"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame from the webcam.")
        break

    # Perform object detection on the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    recognized_sign = None  # Variable to store the recognized sign text

    for r in results:
        annotator = Annotator(frame)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Get box coordinates in (top, left, bottom, right) format
            c = box.cls
            sign_name = model.names[int(c)]
            annotator.box_label(b, sign_name)
            recognized_sign = sign_name  # Set recognized_sign to the detected sign name

    frame = annotator.result()

    # Check if a sign was recognized, and if so, convert it to speech
    if recognized_sign:
        tts = gTTS(text=recognized_sign, lang='en')  # You can specify the language
        tts.save("output.mp3")  # Save the speech to a file

        # Use subprocess.Popen to play the speech with Windows Media Player
        subprocess.Popen([wmplayer_path, "output.mp3"])

    # Display the frame with object detection results
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
