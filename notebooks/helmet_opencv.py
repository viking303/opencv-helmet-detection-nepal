"# OpenCV real-time helmet detection" 
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Draw predictions
    annotated_frame = results[0].plot()

    cv2.imshow("Helmet Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()