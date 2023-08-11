import pickle 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import cv2

# Load an image from a file



from torchvision.models.densenet import densenet201

def combine_predictions(outputs):
    # Apply softmax to the outputs
    # softmax = nn.Softmax(dim=1)
    # probabilities = softmax(outputs)
    # Perform majority voting or averaging
    combined_predictions = torch.mean(outputs, dim=0)
    return combined_predictions

img_path = 'C:/Users/Rahima Haroon/Downloads/img_plant.jpg'
image = Image.open(img_path)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image = transform(image).unsqueeze(0)

# Display the image
# image.show()
class DENN(nn.Module):
    def __init__(self, models):
        super(DENN, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            if model == inception_model:
                outputs.append(output.logits)
            else:
                outputs.append(output)
        outputs = torch.stack(outputs)
        combined_predictions = combine_predictions(outputs)
        return combined_predictions

# Assuming you have defined your models elsewhere

# denn_model = denn_model.to(device)


with open('models.pkl', 'rb') as f:
    loaded_models=pickle.load(f)

print(len(loaded_models))


vgg16_model = loaded_models['vgg16']
inception_model = loaded_models['inception']
densenet_model = loaded_models['densenet']
densenet201_model = loaded_models['densenet201']
resnet_model = loaded_models['resnet']
mobilenet_model = loaded_models['mobilenet']

denn_model = DENN([ densenet_model, resnet_model,mobilenet_model,densenet201_model])


# with open('leaf_mold.txt') as f:
#     content = f.read()


# # print(content)

# a = content.split('\n\n')
# b = a[2].split('What to Do')
# c = b[1].split('\n')
# print(c, sep="\n")
with torch.no_grad():
    output = denn_model(image)
print(output)