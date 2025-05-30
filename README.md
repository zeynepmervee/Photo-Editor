# Image Processing Application

A powerful desktop application for image processing and manipulation built with Python, PyQt5, and OpenCV. This application provides a user-friendly interface for performing various image processing operations with real-time preview capabilities.

## Features

### Image Filters
- Edge Detection (Sobel)
- Basic Filters (Average, Median)
- Image Enhancement (Sharpen, Blur)
- Image Rotation (90°, 180°)
- Image Mirroring (Horizontal, Vertical)

### Morphological Operations
- Dilation
- Erosion
- Skeletonization
- Center of Gravity Calculation

### Histogram Operations
- Histogram Equalization
- Contrast Stretching
- Real-time Histogram Display

### Thresholding
- Manual Threshold Adjustment
- Otsu's Method
- Kapur's Method

## Requirements

- Python 3.x
- PyQt5 >= 5.15.0
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- scikit-image >= 0.21.0
- Matplotlib >= 3.7.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/photo-editor.git
cd photo-editor
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python imageApp.py
```

2. Open an image using the "File" menu or drag and drop an image into the application.

3. Use the various control panels on the right side to:
   - Apply different filters and effects
   - Perform morphological operations
   - Adjust histogram and contrast
   - Apply thresholding operations

4. The left panel shows the original image and its histogram, while the right panel displays the processed image.

5. Save the processed image using the "Save As" button.

## Interface

The application features a clean and intuitive interface with:
- Original image display with histogram
- Processed image preview
- Multiple control panels for different operations
- Real-time updates of processed images
- Menu bar for file operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source. 