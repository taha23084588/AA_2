import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/upload'
app.config['RESULT_FOLDER'] = './static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the pre-trained model (replace with your own model path)
model = load_model('lenet5_model.h5')

# Function to apply Grad-CAM
def apply_gradcam(model, img, last_conv_layer_name):
    grad_model = Model(inputs=model.input, 
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))
        top_pred_idx = tf.argmax(predictions[0])
        top_class_output = predictions[:, top_pred_idx]
    
    grads = tape.gradient(top_class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Normalize heatmap
    return heatmap

# Function to plot the Grad-CAM results
def plot_gradcam_full(img, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(tf.expand_dims(heatmap, axis=-1), (img.shape[0], img.shape[1]))
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()
    heatmap_colormap = plt.cm.jet(heatmap_resized)[:, :, :3]  # Get RGB channels
    superimposed_img = heatmap_colormap * alpha + img
    superimposed_img = superimposed_img / np.max(superimposed_img)  # Normalize
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Superimposed Image")
    plt.imshow(superimposed_img)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Function to apply LIME
def apply_lime(model, image_path):
    # Load and preprocess image for LIME
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    
    # Create a LimeImageExplainer instance
    explainer = lime_image.LimeImageExplainer()
    
    # Explain the model prediction for the image
    explanation = explainer.explain_instance(
        img_array.astype('double'), 
        model.predict, 
        top_labels=5, 
        hide_color=0, 
        num_samples=1000
    )
    
    # Get the image explanation
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundaries = mark_boundaries(temp / 2 + 0.5, mask)
    
    # Save the image with boundaries
    lime_output_path = os.path.join(app.config['RESULT_FOLDER'], 'limemain', 'lime_explanation.jpg')
    plt.imshow(img_boundaries)
    plt.axis('off')
    plt.savefig(lime_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return lime_output_path

# Load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img_array, img

# Function to generate Grad-CAM result and save it
def generate_grad_cam(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    last_conv_layer_name = "conv2d_100"  # Use the correct conv layer
    heatmap = apply_gradcam(model, img_array, last_conv_layer_name)

    # Plot and save results
    heatmap_resized = tf.image.resize(tf.expand_dims(heatmap, axis=-1), (img_array.shape[0], img_array.shape[1]))
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()
    heatmap_colormap = plt.cm.jet(heatmap_resized)[:, :, :3]  # Get RGB channels
    superimposed_img = heatmap_colormap * 0.4 + img_array
    superimposed_img = superimposed_img / np.max(superimposed_img)  # Normalize

    # Save the images
    grad_cam_output_path = os.path.join(app.config['RESULT_FOLDER'], 'gradmain', 'grad_cam_superimposed.jpg')
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.savefig(grad_cam_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return grad_cam_output_path, heatmap_resized

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load and preprocess the image
            img_array, img_original = load_and_preprocess_image(file_path)

            # Make prediction with the model
            predictions = model.predict(np.expand_dims(img_array, axis=0))
            predicted_class = np.argmax(predictions[0])
            predicted_prob = predictions[0][predicted_class]

            # Adjust predicted class based on probability
            if predicted_prob > 0.5:
                predicted_class = 1
            else:
                predicted_class = 0

            # Generate Grad-CAM
            grad_cam_result_path, heatmap = generate_grad_cam(file_path, model)

            # Generate LIME explanation
            lime_result_path = apply_lime(model, file_path)

            # Serve static files (uploaded images, Grad-CAM, LIME)
            uploaded_image_url = f'/static/upload/{filename}'
            grad_cam_image_url = f'/static/results/gradmain/grad_cam_superimposed.jpg'
            lime_image_url = f'/static/results/limemain/lime_explanation.jpg'

            return render_template(
                'index.html',
                uploaded_image=uploaded_image_url,
                predicted_class=predicted_class,
                predicted_prob=predicted_prob,
                grad_cam_image=grad_cam_image_url,
                lime_image=lime_image_url
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
