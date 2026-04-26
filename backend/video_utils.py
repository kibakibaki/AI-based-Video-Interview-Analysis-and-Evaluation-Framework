# video_utils.py
import cv2
import os

def extract_frames(video_path, output_folder):
    """Extract all frames from a video and save as images."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_id += 1

    cap.release()
    print(f"Saved {frame_id} frames.")


def detect_faces(input_folder, output_folder, draw_rectangle=True):
    """Detect faces in all images of input_folder and save results to output_folder."""
    os.makedirs(output_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for filename in os.listdir(input_folder):
        if not filename.endswith(".jpg"):
            continue

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            if draw_rectangle:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

    print("Face detection completed")