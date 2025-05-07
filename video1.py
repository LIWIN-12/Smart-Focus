import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import time
import face_recognition
import os

# Path to input video
VIDEO_PATH = "sample3.mp4"  
OUTPUT_PATH = "output_video3.mp4"

# Load YOLOv8 behavior detection model
#behavior_model = YOLO("trainedmodel6.pt")  
#behavior_model.to('cuda')  # running on GPU

behavior_model = YOLO("trainedmodel6.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
behavior_model.to(device)

# Load Training Images for Face Recognition
path = 'Training_images'
images = []
classNames = []
encodeListKnown = []

for file in os.listdir(path):
    img_path = os.path.join(path, file)
    if not (file.lower().endswith(('.jpg', '.png', '.jpeg'))):
        continue 
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes: 
            images.append(img)
            classNames.append(os.path.splitext(file)[0])
            encodeListKnown.append(encodes[0])

print(f"✅ Loaded {len(classNames)} known faces:", classNames)

# Function to mark attendance
def markAttendance(name):
    filename = 'Attendance.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,Time\n")  
    with open(filename, 'r+') as f:
        lines = f.readlines()
        nameList = [line.split(',')[0] for line in lines]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')
            print(f"✅ Attendance marked for {name} at {dtString}")

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ ERROR: Could not open video file {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer with fallback codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Check if VideoWriter is opened successfully
if not out.isOpened():
    print("❌ Error initializing video writer. Trying a different codec.")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened(): 
        print("❌ Failed to initialize VideoWriter with both mp4v and XVID codecs.")
        exit()

attendance = {}
student_actions = {}

# Define behavior categories
attentive_behaviors = ["upright", "raise_head", "hand-raising", "reading", "writing"]
non_attentive_behaviors = ["Using_phone", "bend", "bow_head", "phone", "sleep", "turn_head"]

# Create the window before resizing it
cv2.namedWindow("Classroom Monitoring", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO behavior detection
    behavior_results = behavior_model(frame)
    detected_labels = []
    student_boxes = []
    for result in behavior_results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()
            label = behavior_model.names[int(bbox.cls[0].cpu().numpy())]
            detected_labels.append(label)
            student_boxes.append([x1, y1, x2, y2])

    # Face Recognition: Loop through all faces detected in the frame
    imgS = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis) if min(faceDis) < 0.5 else -1
        if matchIndex != -1 and matches[matchIndex]:
            name = classNames[matchIndex].upper()
            attendance[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            y1, x2, y2, x1 = [v * 2 for v in faceLoc]  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            markAttendance(name)

    # Process behavior for each detected person
    for i, box in enumerate(student_boxes):
        x1, y1, x2, y2 = box
        label = detected_labels[i]
        
        for student in attendance.keys():
            if student not in student_actions:
                student_actions[student] = {"attentive": 0, "non_attentive": 0}
            if label in attentive_behaviors:
                student_actions[student]["attentive"] += 1
            elif label in non_attentive_behaviors:
                student_actions[student]["non_attentive"] += 1
                
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Resize the window before displaying
    cv2.resizeWindow("Classroom Monitoring", 640, 480)  

    # Show the processed frame
    cv2.imshow("Classroom Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the processed frame to the video file
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# Generate Attendance Summary and save to CSV
attendance_data = []
for student, actions in student_actions.items():
    attentive = actions["attentive"]
    non_attentive = actions["non_attentive"]
    total_actions = attentive + non_attentive
    attendance_ratio = (attentive / total_actions) * 100 if total_actions > 0 else 0
    status = "Present" if attendance_ratio >= 50 else "Absent"
    attendance_data.append([student, attentive, non_attentive, attendance_ratio, status])

columns = ["Student Name", "Attentive Actions", "Non-Attentive Actions", "Attendance Ratio (%)", "Attendance Status"]
df = pd.DataFrame(attendance_data, columns=columns)
df.to_csv("attendance_sum.csv", index=False)

print("📊 Attendance summary saved as attendance_sum.csv")
print(f"🎥 Processed video saved as {OUTPUT_PATH}")
print(f"Using device: {behavior_model.device}")
