import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "static", "uploads") 

RESNET_PATH = os.path.join(MODEL_DIR, "pneumonia_resnet50.h5")
DENSENET_PATH = os.path.join(MODEL_DIR, "densenet_finetuned_model.h5")
MOBILENET_PATH = os.path.join(MODEL_DIR, "mobilenet_pneumonia_finetuned.h5")

os.makedirs(UPLOAD_DIR, exist_ok=True)


print("Loading models...")
resnet_model = load_model(RESNET_PATH)
densenet_model = load_model(DENSENET_PATH)
mobilenet_model = load_model(MOBILENET_PATH)
print("Models loaded successfully")


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or invalid path")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def generate_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

     
        if isinstance(predictions, list):
            predictions = predictions[0]

        
        loss = predictions[:, 0] if predictions.shape[-1] == 1 else predictions[:, 1]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap  


def save_gradcam(img_path, heatmap, output_filename):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    out_path = os.path.join(UPLOAD_DIR, output_filename)
    cv2.imwrite(out_path, superimposed)
    return f"uploads/{output_filename}"  


def predict_all(img_path):
    img = preprocess_image(img_path)

    
    resnet_pred = resnet_model.predict(img)[0]
    densenet_pred = densenet_model.predict(img)[0]
    mobilenet_pred = mobilenet_model.predict(img)[0][0]

    resnet_prob = float(resnet_pred[1]) if len(resnet_pred) > 1 else float(resnet_pred[0])
    densenet_prob = float(densenet_pred[1]) if len(densenet_pred) > 1 else float(densenet_pred[0])
    mobilenet_prob = float(mobilenet_pred)

    avg_prob = (resnet_prob + densenet_prob + mobilenet_prob) / 3
    label = "PNEUMONIA" if avg_prob >= 0.5 else "NORMAL"

    
    res_heatmap = generate_gradcam(resnet_model, img, "conv5_block3_out")
    dense_heatmap = generate_gradcam(densenet_model, img, "conv5_block16_concat")
    mobile_heatmap = generate_gradcam(mobilenet_model, img, "conv_pw_13_relu")  

    label_confidence = avg_prob if label == "PNEUMONIA" else 1 - avg_prob
    return {
        "resnet": resnet_prob,
        "densenet": densenet_prob,
        "mobilenet": mobilenet_prob,
        "average": avg_prob,
        "label": label,
        "label_confidence": label_confidence,
        "gradcam_paths": {
            "resnet": save_gradcam(img_path, res_heatmap, "resnet_gradcam.jpg"),
            "densenet": save_gradcam(img_path, dense_heatmap, "densenet_gradcam.jpg"),
            "mobilenet": save_gradcam(img_path, mobile_heatmap, "mobilenet_gradcam.jpg"),
        }
    }


if __name__ == "__main__":
    test_image = os.path.join(BASE_DIR, "test.jpeg")
    result = predict_all(test_image)
    print(result)
