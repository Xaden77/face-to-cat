import cv2
import csv
import os
from mediapipe.python.solutions import face_mesh as face_mesh_module

label = input("Enter expression label (e.g. happy, neutral, surprised): ")

face_mesh = face_mesh_module.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opened")
    exit()

os.makedirs("data", exist_ok=True)
file_path = "data/face_data.csv"

with open(file_path, "a", newline="") as f:
    writer = csv.writer(f)
    print("Press 's' to save a frame, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        row = None
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                row = []
                for lm in face_landmarks.landmark:
                    row.extend([lm.x, lm.y])
                row.append(label)

                cv2.putText(
                    frame,
                    f"Label: {label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Collect Face Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and row is not None:
            writer.writerow(row)
            print("Saved frame")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
