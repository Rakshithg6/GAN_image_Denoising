import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import google.generativeai as genai
import io

# Load the Pre-trained Model (Using a lightweight GAN)
class SimpleGAN(torch.nn.Module):
    def __init__(self):
        super(SimpleGAN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# Load Model (Simulated pre-trained weights)
model = SimpleGAN()
model.load_state_dict(torch.load("gan_denoiser.pth", map_location=torch.device('cpu')))
model.eval()

# Set Up Google Gemini API
genai.configure(api_key="AIzaSyDwlS39w8DpJDCNiLX2QAkxRLekBLVXS-8")

def denoise_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        denoised_image = model(image).squeeze(0)
    
    return transforms.ToPILImage()(denoised_image)

def get_ai_suggestions():
    prompt = "How can I improve image denoising using GANs?"
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text

# Streamlit UI
st.title("GAN-Based Image Denoiser")
st.sidebar.header("Upload an Image")

uploaded_file = st.sidebar.file_uploader("Choose a noisy image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Denoise Image"): 
        denoised = denoise_image(image)
        st.image(denoised, caption="Denoised Image", use_column_width=True)
        
        st.subheader("AI Suggestions for Denoising")
        st.write(get_ai_suggestions())
