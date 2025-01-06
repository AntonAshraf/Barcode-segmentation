# Barcode Detector

This repository contains a barcode detection and recognition system using deep learning models. The system is designed to detect barcodes in images and extract the encoded information. It leverages the power of YOLO for object detection and SAM2 for image segmentation.

## Features

- **Barcode Detection**: Detects barcodes in images using YOLO.
- **Barcode Segmentation**: Segments the detected barcodes using SAM2.
- **Barcode Decoding**: Decodes the segmented barcodes to extract the encoded information using pyzbar.
- **Web Interface**: Provides a user-friendly web interface for uploading images and viewing results.

## Screenshots

|![Screenshot 2025-01-06 230957](https://github.com/user-attachments/assets/d45c228b-0779-451e-9c11-159ffa20d5ad) | ![Screenshot 2025-01-06 231150](https://github.com/user-attachments/assets/840424cb-1ecc-457f-85dc-78cdeafaa872)|
|:--:|:--:|

## Pipeline

![pipeline](https://github.com/user-attachments/assets/5a8c4da8-9fb0-4de7-99f5-3ba728d493ab)

1. **Image Upload**: Users upload an image containing barcodes via the web interface.
2. **Barcode Detection**: The YOLO model detects the barcodes in the uploaded image.
3. **Barcode Segmentation**: The SAM2 model segments the detected barcodes.
4. **Preprocessing**: Apply grayscale and binarization to the cut-out barcode.
5. **Barcode Decoding**: The segmented barcodes are decoded to extract the encoded information.
6. **Result Display**: The original image, processed images, and decoded barcode information are displayed on the web interface.


## Installation

Follow these steps to set up the project:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/barcode-detector.git
    cd barcode-detector
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the pre-trained models**:
    ```sh
    ./checkpoints/download_ckpts.sh
    ```


## Usage

1. **Run the Flask app**:
    ```sh
    python main.py
    ```

2. **Open your browser** and navigate to `http://127.0.0.1:5000`.

3. **Upload an image** containing barcodes and view the results.

## Directory Structure

```
barcode-detector/
├── checkpoints/          # Directory for storing model checkpoints
├── sam2/                 # SAM2 model implementation
├── static/               # Static files for the web interface
├── templates/            # HTML templates for the web interface
├── test images/          # Sample images for testing the system
├── utils/                # Utility scripts
├── YOLO/                 # YOLO model implementation & weights
├── main.py               # Main script to run the Flask app
├── README.md             # This README file
└── requirements.txt      # Python dependencies
```

## Dataset

The dataset used to fine-tune the YOLO model was obtained from [Roboflow](https://universe.roboflow.com/my-workspace-n464v/barcode-detection-mziov).

It contains images of barcodes with corresponding annotations in YOLO format.
The dataset was split into training and validation sets for training the model with heavy data augmentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
