import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import requests
from io import BytesIO

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load an image
img_path = './car.jpg'  # Replace with your local image path
img = Image.open(img_path)
img = img.convert("RGB")

# Transform the image
img_tensor = F.to_tensor(img).unsqueeze(0)

# Perform inference
with torch.no_grad():
    predictions = model(img_tensor)

# Get the predictions
pred_classes = predictions[0]['labels']
pred_boxes = predictions[0]['boxes']
pred_scores = predictions[0]['scores']

# Convert the image for drawing
draw = ImageDraw.Draw(img)

# Draw bounding boxes and labels
for i, score in enumerate(pred_scores):
    if score > 0.5:  # Confidence threshold
        box = pred_boxes[i].tolist()
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{pred_classes[i].item()}", fill="red")

# Show the image with bounding boxes
img.show()

# Save the image with bounding boxes
img.save('result.jpg')
