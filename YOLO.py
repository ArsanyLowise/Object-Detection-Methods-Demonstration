import torch
from PIL import Image
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Loading the image
img_path = './car.jpg'  
img = Image.open(img_path)
img = img.convert("RGB")

# Measure inference time
start_time = time.time()  
results = model(img)
end_time = time.time()  
inference_time = end_time - start_time  

# Print inference time
print(f"Inference Time: {inference_time:.4f} seconds")

# Print and show results
results.print()  
results.show()   
results.save()   
