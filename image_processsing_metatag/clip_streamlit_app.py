import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel

# Set device to CPU only
device = torch.device("cpu")

# Load the model and processor (cached)
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_clip_model()

# UI title and description
st.title("🧠 Poster vs Business Card Classifier")
st.markdown("Classify your image using **OpenAI CLIP** zero-shot model (forced CPU mode). Choose to upload via **URL** or provide a **local path**.")

# Choose input method
input_mode = st.radio("Select image input method:", ["🌐 Image URL", "📁 Local Path"])

image = None

if input_mode == "🌐 Image URL":
    image_url = st.text_input("Paste Image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"⚠️ Could not load image from URL: {e}")

elif input_mode == "📁 Local Path":
    local_path = st.text_input("Enter local image path:")
    if local_path:
        try:
            image = Image.open(local_path).convert("RGB")
        except Exception as e:
            st.error(f"⚠️ Could not load image from path: {e}")

# If valid image is loaded, classify
if image:
    st.image(image, caption="🖼️ Input Image", use_column_width=True)

    labels = [
        "a photo of a business card with contact details",
        "a photo of a promotional poster with event info"
    ]

    # Ensure tensors are created on CPU
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    st.subheader("🔍 Prediction Results:")
    for label, prob in zip(labels, probs[0]):
        st.write(f"- **{label.capitalize()}**: {prob.item()*100:.2f}%")

    final_pred = labels[probs.argmax()]
    st.success(f"✅ Final Verdict: **{final_pred.upper()}**")
