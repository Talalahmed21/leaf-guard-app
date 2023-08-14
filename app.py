import torch
import streamlit as st 
import numpy as np
import pandas
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import pickle
from torchvision.models.densenet import densenet201

# Setting the page config at the beginning of the script
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌱",
    layout="centered"
)



def combine_predictions(outputs):
    # Apply softmax to the outputs
    # softmax = nn.Softmax(dim=1)
    # probabilities = softmax(outputs)
    # Perform majority voting or averaging
    combined_predictions = torch.mean(outputs, dim=0)
    return combined_predictions

class DENN(nn.Module):
    def __init__(self, models):
        super(DENN, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            output = model(x)
            # outputs.append(output.logits)
        outputs.append(output)
        outputs = torch.stack(outputs)
        combined_predictions = combine_predictions(outputs)
        return combined_predictions

# Find the correct key for the model in the checkpoint dictionary
# Once you identify the correct key name, replace 'model_key' with that name
with open('models.pkl', 'rb') as f:
    loaded_models=pickle.load(f)
# # Now you can work with the loaded model on the CPU


# Retrieve the models and checkpoints
densenet_model = loaded_models['densenet']
densenet201_model = loaded_models['densenet201']
resnet_model = loaded_models['resnet']
mobilenet_model = loaded_models['mobilenet']

model = DENN([ densenet_model, resnet_model,mobilenet_model,densenet201_model])

model.eval()  # Set the model to evaluation mode

# Define the class labels (if available)
class_labels = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Background_without_leaves', 'Cherry_Powdery_mildew', 'Corn_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_Common_rust', 'Blueberry_healthy', 'Cherry_healthy', 'Grape_healthy', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Corn_healthy', 'Corn_Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Peach_healthy', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Pepper,_bell_Bacterial_spot', 'Soybean_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Strawberry_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Tomato_Tomato_mosaic_virus', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato__Septoria_leaf_spot']

# Function to perform inference on the input image
def predict_disease(image):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

              
 # Get the top predicted class and its probability

        _, predicted_idx = torch.max(probabilities, 1)
        predicted_label = class_labels[predicted_idx.item()]
        prediction_percentage = probabilities[0, predicted_idx].item() * 100
# np.argmax(probabilities) *100
    return predicted_label, prediction_percentage

# Create the Streamlit app

def main():
# Add your logo image here
    logo_image = ".\logo.jpeg"
    
    # Set a custom CSS style for the header
    st.markdown(
        f"""
        <style>
            .header {{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Create a header with the logo
    # st.markdown('<div class="header"><img src="logo.jpeg" alt="Logo" /></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image('logo.png')

    with col3:
        st.write(' ')

    st.title("Plant Leaf Disease Detection")
   
    # Add an option to upload an image or use the camera
    image_option = st.radio("Select Image Source:",
                            ("Upload Image", "Use Camera"))

    if image_option == "Upload Image":
        # Option to upload an image
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])
        
    elif image_option == "Use Camera":
        # Capture a single image from the camera
        uploaded_file = st.camera_input("Webcam image")

        

    if uploaded_file is not None:
            
            # Display the uploaded image
            image = Image.open(uploaded_file)
            if image_option == "Upload Image":
                st.image(image, caption="Uploaded Image", width=300)


            # Perform prediction on the uploaded image
            predicted_label = predict_disease(image)
            predicted_label, prediction_percentage = predict_disease(image)
            st.write(f"Predicted Disease: {predicted_label}")
            # st.write(f"Prediction Percentage: {prediction_percentage:.2f}%")
            # st.write(predicted_label)
            # Additional information for "Apple__Apple_scab"
            if (predicted_label == "Apple__Apple_scab") or (predicted_label == "Apple__Apple_scab"):

                with open('Apple_scab.txt') as f:
                    content = f.read()


                a = content.split('\n\n')

                b = a[0].split('Apple scab Disease')
                c = b[1].split('\n')
                st.header('Apple Scab Disease')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")


            elif predicted_label == 'Apple_Black_rot':

                with open('Apple_Black_rot.txt') as f:
                    content = f.read()


                a = content.split('\n\n')

                b = a[0].split('Apple Black Rot Disease')
                c = b[1].split('\n')
                st.header('Apple Black Rot')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Apple_Cedar_apple_rust':

                with open('Apple_cedar_apple_rust.txt') as f:
                    content = f.read()


                a = content.split('\n\n')

                b = a[0].split('Apple Cedar Apple Rust Disease')
                c = b[1].split('\n')
                st.header('Apple Cedar Apple Rust')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
             
            elif predicted_label == 'Background_without_leaves':
                st.write("No leaf detected")

           

            elif predicted_label == 'Cherry_Powdery_mildew':

                with open('Cherry_powdery_mildew.txt') as f:
                    content = f.read()


                a = content.split('\n\n')

                b = a[0].split('Cherry Powdery Mildew Disease')
                c = b[1].split('\n')
                st.header('Cherry Powdery Mildew')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Corn_Cercospora_leaf_spot Gray_leaf_spot':

                with open('Corn_Cercospora_leaf_spot Gray_leaf_spot.txt') as f:
                    content = f.read()


                a = content.split('\n\n')

                b = a[0].split('Corn Cercospora Leaf Spot (Gray Leaf Spot) Disease')
                c = b[1].split('\n')
                st.header('Corn Cercospora Leaf Spot (Gray Leaf Spot)')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Corn_Common_rust':

                with open('Corn_common_rust.txt') as f:
                    content = f.read()


                a = content.split('\n\n')

                b = a[0].split('Corn Common Rust Disease')
                c = b[1].split('\n')
                st.header('Corn Common Rust')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)':

                with open('Grape_Leaf_blight(Isariopsis_Leaf_Spot).txt') as f:
                    content = f.read()


                a = content.split('\n\n')

                b = a[0].split('Grape Leaf Blight (Isariopsis Leaf Spot) Disease')
                c = b[1].split('\n')
                st.header('Grape Leaf Blight (Isariopsis Leaf Spot)')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Orange__Haunglongbing(Citrus_greening)':

                with open('Orange__Haunglongbing(Citrus_greening).txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Orange Haunglongbing (Citrus Greening) Disease')
                c = b[1].split('\n')
                st.header('Orange Haunglongbing (Citrus Greening)')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Peach__Bacterial_spot':

                with open('Peach__Bacterial_spot.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Peach Bacterial Spot Disease')
                c = b[1].split('\n')
                st.header('Peach Bacterial Spot')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Corn_Northern_Leaf_Blight':

                with open('Corn_Northern_Leaf_Blight.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Corn Northern Leaf Blight Disease')
                c = b[1].split('\n')
                st.header('Corn Northern Leaf Blight')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Grape_Black_rot':

                with open('Grape_Black_rot.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Grape Black Rot Disease')
                c = b[1].split('\n')
                st.header('Grape Black Rot')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Grape_Esca(Black_Measles)':

                with open('Grape_Esca(Black_Measles).txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Grape Esca (Black Measles) Disease')
                c = b[1].split('\n')
                st.header('Grape Esca')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Potato__Late_blight':

                with open('Potato__Late_blight.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Potato Late Blight Disease')
                c = b[1].split('\n')
                st.header('Potato Late Blight')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Potato_Early_blight':

                with open('Potato_Early_blight.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Potato Early Blight Disease')
                c = b[1].split('\n')
                st.header('Potato Early Blight')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Pepper,_bell_Bacterial_spot':

                with open('Pepper,_bell_Bacterial_spot.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Pepper Bell Bacterial Spot Disease')
                c = b[1].split('\n')
                st.header('Pepper Bell Bacterial Spot')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Tomato_Bacterial_spot':

                with open('Tomato_Bacterial_spot.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Bacterial Spot Disease')
                c = b[1].split('\n')
                st.header('Tomato Bacterial Spot')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Tomato_Early_blight':

                with open('Tomato_Early_blight.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Early Blight Disease')
                c = b[1].split('\n')
                st.header('Tomato Early Blight')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Squash_Powdery_mildew':

                with open('Squash_Powdery_mildew.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Squash Powdery Mildew Disease')
                c = b[1].split('\n')
                st.header('Squash Powdery Mildew')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Strawberry_Leaf_scorch':

                with open('Strawberry_Leaf_scorch.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Strawberry Leaf Scorch Disease')
                c = b[1].split('\n')
                st.header('Strawberry Leaf Scorch')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Tomato_Tomato_mosaic_virus':

                with open('Tomato_Tomato_mosaic_virus.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Mosaic Virus Disease')
                c = b[1].split('\n')
                st.header('Tomato Mosaic Virus')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Tomato_Spider_mites Two-spotted_spider_mite':

                with open('Tomato_Spider_mites Two-spotted_spider_mite.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Spider Mites (Two-Spotted Spider Mite) Disease')
                c = b[1].split('\n')
                st.header('Tomato Spider Mites')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Tomato_Target_Spot':

                with open('Tomato_Target_Spot.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Target Spot Disease')
                c = b[1].split('\n')
                st.header('Tomato Target Spot')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Tomato_Tomato_Yellow_Leaf_Curl_Virus':

                with open('Tomato_Yellow_Leaf_Curl_Virus.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Yellow Leaf Curl Virus Disease')
                c = b[1].split('\n')
                st.header('Tomato Yellow Leaf Curl Virus')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Tomato_Late_blight':

                with open('Tomato_Late_blight.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Late Blight Disease')
                c = b[1].split('\n')
                st.header('Tomato Late Blight')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")
            
            elif predicted_label == 'Tomato_Leaf_Mold':

                with open('Tomato_Leaf_Mold.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Leaf Mold Disease')
                c = b[1].split('\n')
                st.header('Tomato Leaf Mold')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif predicted_label == 'Tomato__Septoria_leaf_spot':

                with open('Tomato__Septoria_leaf_spot.txt') as f:
                    content = f.read()

                a = content.split('\n\n')

                b = a[0].split('Tomato Septoria Leaf Spot Disease')
                c = b[1].split('\n')
                st.header('Tomato Septoria Leaf Spot')
                for line in c[1:]:
                    st.write(line)


                b = a[1].split('Possible Causes')
                c = b[1].split('\n')
                
                st.header('Possible Causes')
                for line in c[1:]:
                    st.write(f"- {line}")

                b = a[2].split('Actions Required')
                c = b[1].split('\n')

                st.header('Actions Required')
                for line in c[1:]:
                    st.write(f"- {line}")

            elif (predicted_label == 'Apple_healthy') or (predicted_label == 'Blueberry_healthy') or (predicted_label == 'Cherry_healthy') or (predicted_label == 'Grape_healthy') or (predicted_label == 'Corn_healthy') or (predicted_label == 'Potato_healthy') or (predicted_label == 'Raspberry_healthy') or (predicted_label == 'Peach_healthy') or (predicted_label == 'Pepper,_bell_healthy') or (predicted_label == 'Soybean_healthy') or (predicted_label == 'Tomato_healthy') or (predicted_label == 'Strawberry_healthy'):
                st.markdown("**:black[Congratulations! Your plant appears to be healthy.]**")

            st.write("While your plant disease detection app is a valuable tool, it's essential to remember that its predictions are based on visual similarities. For precise results and effective treatment, consider seeking guidance from experts and conducting proper laboratory tests to confirm the presence of any disease")
            st.write("Thank you for using the Plant Leaf Disease Detection app!")

if __name__ == "__main__":
    main()

