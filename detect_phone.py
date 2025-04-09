import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    results = model(frame, conf=0.4)  # lower confidence threshold if needed
    


    # Draw boxes
    annotated_frame = results[0].plot( )

    cv2.imshow("Phone Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
