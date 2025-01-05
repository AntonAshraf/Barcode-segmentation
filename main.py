import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.show_sam import show_masks, show_points
from matplotlib import pyplot as plt


sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
print("Loading SAM2 Model...")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
print("Loading SAM2 Model...")
sam2_model = build_sam2(model_cfg, sam2_checkpoint)
print("SAM2 Model loaded successfully!x")

predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 Model loaded successfully!")

# Initialize Flask app
app = Flask(__name__)

# YOLO Model for Barcode Detection
model = YOLO('YoloV8s30-best.pt')
print("YOLO Model loaded successfully!")
# Upload folder
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def get_box_center(box):
  center_x = int((box[0] + box[2]) / 2)
  center_y = int((box[1] + box[3]) / 2)
  return np.array([[center_x, center_y]])

# Function to preprocess the image for OCR
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

# Function to read barcode
def read_barcode(cropped_image):
    # Decode the barcode using pyzbar
    barcodes = decode(cropped_image)

    barcode_number = None
    for barcode in barcodes:
        barcode_number = barcode.data.decode('utf-8')
    return barcode_number

# Function to process the image using YOLO and SAM
def process_image(image, output_folder):

    # File paths for output images
    yolo_bbox_path = os.path.join(output_folder, "yolo_bbox.png")
    mask_image_path = os.path.join(output_folder, "mask_image.png")
    # Detect barcode with YOLO
    image = cv2.imread(image)
    results = model(image)
    print("YOLO Results:", results)
    if len(results) > 0:
        # Get the bounding box for the first detected barcode
        bbox = results[0].boxes.xyxy[0]  # x1, y1, x2, y2
        x1, y1, x2, y2 = map(int, bbox)
        input_box = np.array([x1, y1, x2, y2])
        center_point = get_box_center(input_box)
        yolo_bbox = image.copy()
        cv2.rectangle(yolo_bbox, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cv2.imwrite(yolo_bbox_path, yolo_bbox)
        # cv2.imshow("Box", cpimage)
        # print yolo bounding boxs
        # print("YOLO Bounding Box:", input_box)
        # Segment the image using SAM (optional, refine the barcode area)

        input_label = np.array([1])
        # For visualization, show the mask and points
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(center_point, input_label, plt.gca())
        plt.axis('on')
        # plt.show()

        # predict the image using SAM2
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=center_point,
            point_labels=input_label,
            multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        show_masks(rgb_image, masks, scores, point_coords=center_point, input_labels=input_label, borders=False, mask_image_path=mask_image_path)
        
        # Choose the highest score mask
        sorted_ind = np.argsort(scores)[::-1]
        best_mask = masks[sorted_ind][0]

        # Convert the mask to boolean
        best_mask = best_mask.astype(bool)

        # Create a cutout of the image where the mask is applied
        cutout = np.zeros_like(image)
        cutout[best_mask] = image[best_mask]

        # Calculate the size (height and width) of the mask
        mask_height, mask_width = best_mask.shape

        # Find the bounding box of the mask
        y_indices, x_indices = np.where(best_mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Crop the image using the bounding box
        cropped_image = cutout[y_min:y_max+1, x_min:x_max+1]
        cv2.imshow("Cropped Image", cropped_image)

        # You can use the first mask or all masks, depending on your use case

        # Preprocess the cropped image for OCR (grayscale, binarization)
        processed_image = preprocess_image(cutout)
        # cv2.imshow("Processed Image", processed_image)

        # Use pyzbar to read the barcode from the processed image
        barcode_number = read_barcode(cropped_image)
        print("Barcode Number:", barcode_number)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return barcode_number, yolo_bbox_path, mask_image_path
    
    return "No barcode detected"


# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Generate outputs
            bcNumber, yolo_bbox_path, mask_image_path = process_image(
                file_path,
                app.config['OUTPUT_FOLDER']
            )

            return render_template(
                "index.html",
                original_image=file_path,
                yolo_bbox=yolo_bbox_path,
                mask_image=mask_image_path,
                barcode_number=bcNumber,
            )

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
