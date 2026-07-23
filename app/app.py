import streamlit as st
from PIL import Image
from inference import PoultryPredictor

# Page Configuration
st.set_page_config(
    page_title="Poultry Flock Health Triage System",
    page_icon="🐔",
    layout="wide",
)


# Load Predictor
@st.cache_resource
def load_predictor():
    return PoultryPredictor()


predictor = load_predictor()

# Title
st.title("🐔 Poultry Flock Health Triage System")

st.markdown("""
Upload an image of poultry droppings to predict whether it is
**Healthy** or **Unhealthy**.
""")

st.divider()

# Upload Image
uploaded_file = st.file_uploader("Upload image", type=["jpeg", "png", "jpg"])

# Prediction

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Create layout only once
    left_col, right_col = st.columns([1, 1])

    image_placeholder = left_col.empty()
    prediction_placeholder = right_col.empty()

    # Loading placeholders

    with image_placeholder.container():

        st.subheader("Uploaded Image")
        st.info("🖼️ Loading image...")

    with prediction_placeholder.container():

        st.subheader("Prediction")
        st.info("🔍 Analyzing poultry dropping...")

    # Run inference

    with st.spinner("Running model inference..."):
        result = predictor.predict(image)

    # Replace placeholders

    with image_placeholder.container():

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True,
        )

    with prediction_placeholder.container():

        st.subheader("Prediction")

        prediction = result["prediction"]
        confidence = result["confidence"]

        if prediction == "Healthy":
            st.success(f"🟢 {prediction}")
        else:
            st.error(f"🔴 {prediction}")

        st.metric(
            "Confidence",
            f"{confidence:.2%}",
        )

        st.divider()

        st.subheader("Class Probabilities")

        healthy_prob = result["probabilities"]["Healthy"]
        unhealthy_prob = result["probabilities"]["Unhealthy"]

        st.write(f"Healthy — {healthy_prob:.2%}")
        st.progress(healthy_prob)

        st.write(f"Unhealthy — {unhealthy_prob:.2%}")
        st.progress(unhealthy_prob)
