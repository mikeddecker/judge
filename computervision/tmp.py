import cv2
import os
from ultralytics import YOLO


modeltrainround = 7
modelpath = f"/home/miked/code/judge/runs/detect/train{modeltrainround}" if modeltrainround > 0 else f"/home/miked/code/judge/runs/detect/train"
modelpath


# Load the pre-trained YOLO model
model = YOLO(os.path.join(modelpath, "weights", "best.pt"))


# Load image using OpenCV
imagepath = "1405_1061_cropped.jpg"
annotated_imagepath = "1267_292_spectator_reduced.jpg"
image = cv2.imread(imagepath)


# Run detection
results = model(image)  # conf is confidence threshold, adjust if needed

# The model automatically outputs annotated images, but we'll also manually draw if needed:
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[class_id]} {conf:.2f}" if hasattr(model, 'names') else f"ID {class_id} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the annotated image
cv2.imwrite(annotated_imagepath, image)

print("Detection and annotation complete!")
