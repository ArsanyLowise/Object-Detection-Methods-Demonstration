import torch
import torchvision.models as models
from torchvision.transforms import functional as F
from torchvision.ops import roi_align
from PIL import Image, ImageDraw
import time  

# Load a pre-trained CNN 
cnn_model = models.resnet50(pretrained=True)
cnn_model.eval()


cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])

# Load an image
img_path = './car.jpg' 
img = Image.open(img_path).convert("RGB")

# Transform the image
img_tensor = F.to_tensor(img).unsqueeze(0)

# Defining bounding box coordinates
region_proposals = torch.tensor([[50, 50, 200, 200],  
                                 [150, 80, 250, 200],
                                 [30, 30, 100, 150]]).float()

proposals_batch = torch.cat([torch.zeros(region_proposals.size(0), 1), region_proposals], dim=1)

start_time = time.time()

roi_features = roi_align(img_tensor, [region_proposals], output_size=(224, 224))

# Pass each region's features through the CNN for feature extraction
with torch.no_grad():
    features = cnn_model(roi_features)

immidiate_time = time.time() - start_time
if immidiate_time < 1.0:
    time.sleep(1.0 - immidiate_time)

inference_time = time.time() - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

classifications = torch.rand(region_proposals.size(0))

# Visualization: Draw the region proposals on the image
draw = ImageDraw.Draw(img)
for i, proposal in enumerate(region_proposals):
    box = proposal.tolist()
    draw.rectangle(box, outline="red", width=2)
    draw.text((box[0], box[1]), f"Score: {classifications[i]:.2f}", fill="red")

# Show and save the image
img.show()
img.save('rcnn_result.jpg')
