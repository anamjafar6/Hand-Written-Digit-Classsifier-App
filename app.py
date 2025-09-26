import streamlit as st # makes the web app
import numpy as np #handles math & image arrays
import cv2 #image processing(resize, grayscale)
from PIL import Image # image handling, works with uploaded images
from tensorflow.keras.models import load_model #loads the trained mnist model
from streamlit_drawable_canvas import st_canvas #Imports the drawable canvas component for Streamlit which lets you draw digits inside app

#This sets up the web app title, icon, and layout.
st.set_page_config(
    page_title="Digit Recognition App",
    page_icon="🔢",
    layout="wide"
)

# The CSS part changes the background color gradient
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e3f2fd);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# This loads the trained model (mnist_cnn.h5) that recognizes digits.
@st.cache_resource #makes sure the model loads only once, not every time you interact.
def load_cnn_model(): 
    return load_model("mnist_cnn.h5") #loads the saved model file mnist_cnn.h5.

model = load_cnn_model() #stores the loaded model in the variable model so you can call it later.

# ---- Preprocessing Helpers ----
def preprocess_pil_file(file_or_pil_image):  #Defines a function which accepts either a file or a Image.
    if not isinstance(file_or_pil_image, Image.Image):
        img = Image.open(file_or_pil_image) #If the input is not already a PIL Image object, use Image.open() to load it. Otherwise, use it directly.
    else:
        img = file_or_pil_image #PIL = Python Imaging Library .It’s a library that helps you open, edit, and save images in Python.

    img = img.convert('L').resize((28, 28)) # Convert to grayscale & resize
    arr = np.array(img).astype('float32') / 255.0 # Normalize (0–1 scale)

    if arr.mean() > 0.5:  # invert if background is white
        arr = 1.0 - arr

    arr = arr.reshape(1, 28, 28, 1).astype('float32') #Reshape to fit model input.
    return arr, Image.fromarray((arr[0, :, :, 0] * 255).astype('uint8'))

def preprocess_canvas_image(image_data):
    if image_data is None:
        return None, None

    img_uint8 = image_data.astype('uint8')

    if img_uint8.shape[2] == 4:  # RGBA
        alpha = img_uint8[..., 3] / 255.0
        rgb = img_uint8[..., :3].astype('float32')
        white = np.ones_like(rgb) * 255.0
        comp = (rgb * alpha[..., None] + white * (1 - alpha[..., None])).astype('uint8')
        gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) # convert canvas to grayscale

    small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA).astype('float32') / 255.0

    if small.mean() > 0.5:
        small = 1.0 - small

    arr = small.reshape(1, 28, 28, 1).astype('float32')
    display_img = Image.fromarray((small * 255).astype('uint8'))
    return arr, display_img

# ---- Header ----
st.markdown("<h1 style='text-align:center;color:#0D47A1;'>🔢 Handwritten Digit Recognizer</h1>", unsafe_allow_html=True)
st.write("Upload or draw a digit (0–9). The app will preprocess the image and predict the digit.")
st.markdown("---")

# ---- Sidebar ----
st.sidebar.header("📌 Instructions")
st.sidebar.info(
    "• Upload PNG/JPG or draw a digit.  \n"
    "• The app auto-preprocesses (grayscale, resize, normalize, invert if needed).  \n"
    "• Predictions show digit + confidence & probability bar chart."
)
st.sidebar.markdown("---")
st.sidebar.write("👩‍💻 **About**: Built with streamlit❤️ by **Anam Jafar**")
st.sidebar.write("[🔗 LinkedIn](https://www.linkedin.com/in/anam-jafar6/)")

# ---- File Upload ----
uploaded_files = st.file_uploader(
    "📂 Upload digit images (single or multiple):",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("📷 Uploaded Images & Predictions")

    max_cols = 4
    for i in range(0, len(uploaded_files), max_cols):
        row_files = uploaded_files[i:i+max_cols]
        cols = st.columns(len(row_files))
        for j, file in enumerate(row_files):
            arr, display_img = preprocess_pil_file(file)
            pred = model.predict(arr)
            probs = pred[0]
            label = int(np.argmax(probs))
            conf = float(np.max(probs))

            with cols[j]:
                st.image(display_img, caption=f"Pred: {label} ({conf*100:.1f}%)", width=60)
                st.bar_chart(probs)

# ---- Drawing Pad ----
st.subheader("🖌️ Draw your digit here:")
canvas_result = st_canvas(
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result is not None and canvas_result.image_data is not None:
    arr, display_img = preprocess_canvas_image(canvas_result.image_data)
    if arr is not None:
        pred = model.predict(arr)
        probs = pred[0]
        label = int(np.argmax(probs))
        conf = float(np.max(probs))

        st.markdown(
            f"""
            <div style="padding:12px;border-radius:8px;background:#FFF3CD;text-align:center;">
                <h2 style="color:#D32F2F;">🎯 Predicted Digit: {label}</h2>
                <p>Confidence: {conf*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(display_img, caption="Preprocessed (28×28) view", width=60)
        st.bar_chart(probs)

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit & TensorFlow | By <b>Anam Jafar</b></p>",
    unsafe_allow_html=True
)
