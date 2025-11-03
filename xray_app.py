import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Prepare the image for prediction (CNN)
def prepare_image_cnn(image, img_size=150):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.reshape(image, (1, img_size, img_size, 1))
    return image

# Prepare the image for prediction (ResNet)
def prepare_image_resnet(image, img_size=224):
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 1:
            image = np.stack((image,) * 3, axis=-1)
        image = cv2.resize(image, (img_size, img_size))
        image = image / 255.0
        image = np.reshape(image, (1, img_size, img_size, 3))
        return image
    except Exception as e:
        st.error(f"Error during image preparation: {e}")
        return None

# Prepare the image for prediction (VGG16/VGG16)
def prepare_image_vgg16(image, img_size=150):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.reshape(image, (1, img_size, img_size, 3))
    return image

# Load CNN model
@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("pneumonia_model.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load CNN model: {e}")
        return None

# Load ResNet50 model
@st.cache_resource
def load_resnet_model():
    try:
        model = tf.keras.models.load_model("pneumonia_resnet_complete.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load ResNet50 model: {e}")
        return None

# Load VGG16/VGG16 model
@st.cache_resource
def load_vgg16_model():
    try:
        model = tf.keras.models.load_model("model_VGG16.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load VGG16 model: {e}")
        return None

# Prediction functions for each model
def predict_with_cnn(model, image):
    try:
        predictions = model.predict(image)
        confidence = float(predictions[0][0] if predictions[0][0] > 0.5 else 1 - predictions[0][0]) * 100
        condition_index = int(predictions[0][0] > 0.5)
        return True, predictions, confidence, condition_index
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return False, None, None, None

def predict_with_resnet(model, image):
    try:
        predictions = model.predict(image)
        confidence = float(predictions[0][0] if predictions[0][0] > 0.5 else 1 - predictions[0][0]) * 100
        condition_index = int(predictions[0][0] > 0.5)
        return True, predictions, confidence, condition_index
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return False, None, None, None

def predict_with_vgg16(model, image):
    try:
        predictions = model.predict(image)
        confidence = float(predictions[0][0] if predictions[0][0] > 0.5 else 1 - predictions[0][0]) * 100
        condition_index = int(predictions[0][0] > 0.5)
        return True, predictions, confidence, condition_index
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return False, None, None, None

# Sidebar: Model selection
st.sidebar.title("Model Selection")
model_options = ["CNN", "ResNet50", "VGG16"]
selected_model = st.sidebar.radio("Choose a model:", model_options)
st.sidebar.write(f"Selected Model: {selected_model}")

# Main application
def main():
    st.title("X-ray Image Analysis")
    st.write("Upload an X-ray image for analysis.")

    uploaded_file = st.file_uploader("Upload X-ray Image (JPEG, PNG):", type=["jpg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if image is not None:
            st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

            try:
                # Prepare the image based on the selected model
                if selected_model == "CNN":
                    prepared_image = prepare_image_cnn(image)
                    model = load_cnn_model()
                    predict_func = predict_with_cnn
                elif selected_model == "ResNet50":
                    prepared_image = prepare_image_resnet(image)
                    model = load_resnet_model()
                    predict_func = predict_with_resnet
                elif selected_model == "VGG16":
                    prepared_image = prepare_image_vgg16(image)
                    model = load_vgg16_model()
                    predict_func = predict_with_vgg16

                if prepared_image is not None and model is not None:
                    if st.button("Process"):
                        st.info("\U0001F50E **Processing started... Please wait.**")
                        success, raw_predictions, confidence, condition_index = predict_func(model, prepared_image)

                        class_labels = ['Class 0', 'Class 1']  # Default class names

                        if success:
                            #st.success("X-ray image successfully analyzed.")
                            #st.write(f"Raw Prediction Values (Probabilities): {raw_predictions}")
                            #st.write(f"Model Confidence (Highest Probability): **{confidence:.2f}%**")
                            #st.write(f"Predicted Class Index: {condition_index}")

                            # Update class names according to the selected model
                            if selected_model == "CNN":
                                class_labels = ['Pneumonia (Class 0)', 'Normal (Class 1)']
                            else:
                                class_labels = ['Normal (Class 0)', 'Pneumonia (Class 1)']
                            
                            st.success("\U00002705 **Analysis Complete!**")
                            
                            # Show prediction confidence
                            st.markdown(f"### **Model Confidence:**")
                            st.markdown(f"<div style='font-size: 24px; color: blue;'>\U0001F4C8 {confidence:.2f}%</div>", unsafe_allow_html=True)

                            predicted_class = class_labels[condition_index]
                            st.markdown(f"### **Predicted Class:**")
                            # Color change based on predicted class
                            if 'Pneumonia' in predicted_class:
                                st.markdown(f"<div style='font-size: 24px; font-weight: bold; color: red;'>\U0001F5F8 {predicted_class}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='font-size: 24px; font-weight: bold; color: green;'>\U0001F5F8 {predicted_class}</div>", unsafe_allow_html=True)
                            """st.markdown(f"<div style='font-size: 24px; font-weight: bold; color: green;'>\U0001F5F8 {predicted_class }</div>", unsafe_allow_html=True)"""

                            # Bar chart plotting
                            # Reverse prediction probabilities
                            # Reverse prediction probabilities

                            if selected_model=="CNN":
                                class_labels = ['Normal (Class 1)', 'Pneumonia (Class 0)']
                                prediction_probabilities = [raw_predictions[0][0], 1 - raw_predictions[0][0]]  # Reverse the first prediction values
                                # Create a bar chart
                                fig, ax = plt.subplots()
                                colors = ['green', 'red']  # Red for Pneumonia, Green for Normal
                                ax.bar(class_labels, prediction_probabilities, color=colors)
                                ax.set_title("Model Raw Prediction")
                                ax.set_xlabel("Class")
                                ax.set_ylabel("Prediction Probability")
                                st.pyplot(fig)
                            else:
                                # Create a bar chart
                                prediction_probabilities = [1 - raw_predictions[0][0], raw_predictions[0][0]]  # Reverse the first prediction values
                                fig, ax = plt.subplots()
                                colors = ['green', 'red']  # Green for Normal, Red for Pneumonia
                                ax.bar(class_labels, prediction_probabilities, color=colors)
                                ax.set_title("Model Raw Prediction")
                                ax.set_xlabel("Class")
                                ax.set_ylabel("Prediction Probability")
                                st.pyplot(fig)


                        else:
                            st.error("Prediction failed. Please try again.")
                else:
                    st.error("Image preparation or model loading failed.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
