import pickle
import torch

loaded_models = torch.load('models.pkl', map_location=torch.device('cpu'))


# Retrieve the models and checkpoints
vgg16_model = loaded_models['vgg16']
inception_model = loaded_models['inception']
densenet_model = loaded_models['densenet']
densenet201_model = loaded_models['densenet201']
resnet_model = loaded_models['resnet']
mobilenet_model = loaded_models['mobilenet']

# Save models into a dictionary
saved_models = {
    'densenet': densenet_model,
    'densenet201': densenet201_model,
    'resnet': resnet_model,
    'mobilenet': mobilenet_model
}

# Save the dictionary using pickle
with open('models2.pkl', 'wb') as f:
    pickle.dump(saved_models, f)