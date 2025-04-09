import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read one frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame)

    # Get all detections (like positions and labels)
    for result in results:
        boxes = result.boxes  # List of detected boxes
        
        for box in boxes:
            # Get the center point (x, y) of the box
            x1, y1, x2, y2 = box.xyxy[0]  # top-left and bottom-right corners
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Get the class label
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Draw a small red dot at the center
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

            # Put the label near the dot
            cv2.putText(frame, label, (x_center + 10, y_center - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the result
    cv2.imshow("Phone Detection (Dot + Label)", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
