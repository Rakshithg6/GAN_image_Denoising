# GAN-Based Image Denoiser

This project implements a simple **GAN-based image denoiser** using PyTorch, along with a **Google Gemini AI integration** to provide suggestions on improving GAN-based denoising techniques. The web app is built using **Streamlit**, providing an easy-to-use interface for image uploading, denoising, and receiving AI-based suggestions.

## Features

- **Image Upload**: Users can upload noisy images in **PNG**, **JPG**, or **JPEG** format.
- **Denoising**: The uploaded image is processed by a pre-trained **GAN model** to remove noise and enhance the image.
- **AI Suggestions**: Powered by **Google Gemini API**, the app generates suggestions on how to improve the image denoising technique.

## Setup and Installation

### Prerequisites

Before running the app, make sure you have the following dependencies installed:

- **Python 3.x**
- **Streamlit**
- **PyTorch**
- **PIL (Python Imaging Library)**
- **Google Generative AI (Gemini)**

### Install Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GAN_image_Denoising.git
   cd GAN_image_Denoising
```
2.Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use 'venv\
```

4.Install the required Python packages:

```bash
pip install -r requirements.txt
```
