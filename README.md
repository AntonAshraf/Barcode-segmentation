# Barcode Detector

This repository contains a barcode detection and recognition system using deep learning models. The system is designed to detect barcodes in images and extract the encoded information. It leverages the power of YOLO for object detection and SAM2 for image segmentation.

## Features

- **Barcode Detection**: Detects barcodes in images using YOLO.
- **Barcode Segmentation**: Segments the detected barcodes using SAM2.
- **Barcode Decoding**: Decodes the segmented barcodes to extract the encoded information.
- **Web Interface**: Provides a user-friendly web interface for uploading images and viewing results.

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

## Pipeline

1. **Image Upload**: Users upload an image containing barcodes via the web interface.
2. **Barcode Detection**: The YOLO model detects the barcodes in the uploaded image.
3. **Barcode Segmentation**: The SAM2 model segments the detected barcodes.
4. **Barcode Decoding**: The segmented barcodes are decoded to extract the encoded information.
5. **Result Display**: The original image, processed images, and decoded barcode information are displayed on the web interface.

![](/pipeline.jpg)
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
├── configs/              # Configuration files for models
├── notebooks/            # Jupyter notebooks for experiments
├── sam2/                 # SAM2 model implementation
├── static/               # Static files for the web interface
├── templates/            # HTML templates for the web interface
├── utils/                # Utility scripts
├── main.py               # Main script to run the Flask app
├── README.md             # This README file
└── requirements.txt      # Python dependencies
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
