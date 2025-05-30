import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QMainWindow, QAction, QMessageBox,
                             QToolBar, QSpinBox, QComboBox, QSlider, QGroupBox, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu, threshold_yen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Image Processing Application")
        self.setGeometry(100, 100, 1200, 800)

        self.image = None
        self.processed_image = None
        self.histogram_figure = None
        self.histogram_canvas = None

        self.initUI()

    def initUI(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create left panel for original image and histogram
        left_panel = QVBoxLayout()
        
        # Original image label
        self.imageLabel = QLabel("Original Image")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setFixedSize(400, 300)
        left_panel.addWidget(self.imageLabel)

        # Histogram canvas
        self.histogram_canvas = FigureCanvas(plt.figure(figsize=(4, 2)))
        left_panel.addWidget(self.histogram_canvas)

        # Create right panel for processed image and controls
        right_panel = QVBoxLayout()
        
        # Processed image label
        self.processedLabel = QLabel("Processed Image")
        self.processedLabel.setAlignment(Qt.AlignCenter)
        self.processedLabel.setFixedSize(400, 300)
        right_panel.addWidget(self.processedLabel)

        # Create control panels
        self.create_filters_panel()
        self.create_morphological_panel()
        self.create_histogram_panel()
        self.create_threshold_panel()

        # Add panels to right layout
        right_panel.addWidget(self.filters_group)
        right_panel.addWidget(self.morphological_group)
        right_panel.addWidget(self.histogram_group)
        right_panel.addWidget(self.threshold_group)

        # Add save button
        self.saveBtn = QPushButton("Save As")
        self.saveBtn.clicked.connect(self.save_image)
        right_panel.addWidget(self.saveBtn)

        # Add panels to main layout
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)

        # Create menu bar
        self.create_menu_bar()

    def create_filters_panel(self):
        self.filters_group = QGroupBox("Filters")
        layout = QGridLayout()

        # Edge detection
        self.sobelBtn = QPushButton("Sobel")
        self.sobelBtn.clicked.connect(self.apply_sobel)
        layout.addWidget(self.sobelBtn, 0, 0)

        self.cannyBtn = QPushButton("Canny")
        self.cannyBtn.clicked.connect(self.apply_canny)
        layout.addWidget(self.cannyBtn, 0, 1)

        # Basic filters
        self.avgBtn = QPushButton("Average")
        self.avgBtn.clicked.connect(self.apply_average_filter)
        layout.addWidget(self.avgBtn, 1, 0)

        self.medianBtn = QPushButton("Median")
        self.medianBtn.clicked.connect(self.apply_median_filter)
        layout.addWidget(self.medianBtn, 1, 1)

        self.sharpBtn = QPushButton("Sharpen")
        self.sharpBtn.clicked.connect(self.apply_sharpening)
        layout.addWidget(self.sharpBtn, 2, 0)

        self.blurBtn = QPushButton("Blur")
        self.blurBtn.clicked.connect(self.apply_blur)
        layout.addWidget(self.blurBtn, 2, 1)

        # Rotation
        self.rotate90Btn = QPushButton("Rotate 90°")
        self.rotate90Btn.clicked.connect(lambda: self.rotate_image(90))
        layout.addWidget(self.rotate90Btn, 3, 0)

        self.rotate180Btn = QPushButton("Rotate 180°")
        self.rotate180Btn.clicked.connect(lambda: self.rotate_image(180))
        layout.addWidget(self.rotate180Btn, 3, 1)

        # Mirroring
        self.hMirrorBtn = QPushButton("Horizontal Mirror")
        self.hMirrorBtn.clicked.connect(lambda: self.mirror_image(True))
        layout.addWidget(self.hMirrorBtn, 4, 0)

        self.vMirrorBtn = QPushButton("Vertical Mirror")
        self.vMirrorBtn.clicked.connect(lambda: self.mirror_image(False))
        layout.addWidget(self.vMirrorBtn, 4, 1)

        self.filters_group.setLayout(layout)

    def create_morphological_panel(self):
        self.morphological_group = QGroupBox("Morphological Operations")
        layout = QGridLayout()

        self.dilateBtn = QPushButton("Dilate")
        self.dilateBtn.clicked.connect(self.apply_dilation)
        layout.addWidget(self.dilateBtn, 0, 0)

        self.erodeBtn = QPushButton("Erode")
        self.erodeBtn.clicked.connect(self.apply_erosion)
        layout.addWidget(self.erodeBtn, 0, 1)

        self.skeletonBtn = QPushButton("Skeletonize")
        self.skeletonBtn.clicked.connect(self.apply_skeletonization)
        layout.addWidget(self.skeletonBtn, 1, 0)

        self.cogBtn = QPushButton("Center of Gravity")
        self.cogBtn.clicked.connect(self.compute_center_of_gravity)
        layout.addWidget(self.cogBtn, 1, 1)

        self.morphological_group.setLayout(layout)

    def create_histogram_panel(self):
        self.histogram_group = QGroupBox("Histogram Operations")
        layout = QGridLayout()

        self.equalizeBtn = QPushButton("Equalize")
        self.equalizeBtn.clicked.connect(self.apply_histogram_equalization)
        layout.addWidget(self.equalizeBtn, 0, 0)

        self.contrastBtn = QPushButton("Contrast Stretch")
        self.contrastBtn.clicked.connect(self.apply_contrast_stretching)
        layout.addWidget(self.contrastBtn, 0, 1)

        self.histogram_group.setLayout(layout)

    def create_threshold_panel(self):
        self.threshold_group = QGroupBox("Thresholding")
        layout = QGridLayout()

        self.thresholdSlider = QSlider(Qt.Horizontal)
        self.thresholdSlider.setRange(0, 255)
        self.thresholdSlider.valueChanged.connect(self.apply_manual_threshold)
        layout.addWidget(self.thresholdSlider, 0, 0, 1, 2)

        self.otsuBtn = QPushButton("Otsu")
        self.otsuBtn.clicked.connect(self.apply_otsu_threshold)
        layout.addWidget(self.otsuBtn, 1, 0)

        self.kapurBtn = QPushButton("Kapur")
        self.kapurBtn.clicked.connect(self.apply_kapur_threshold)
        layout.addWidget(self.kapurBtn, 1, 1)

        self.threshold_group.setLayout(layout)

    def create_menu_bar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("File")

        openAction = QAction("Open Image", self)
        openAction.triggered.connect(self.open_image)
        fileMenu.addAction(openAction)

        exitAction = QAction("Exit", self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

    def open_image(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                "Image Files (*.png *.jpg *.bmp)")
        if fileName:
            self.image = cv2.imread(fileName)
            self.display_image(self.image, self.imageLabel)
            self.update_histogram(self.image)

    def display_image(self, img, label, is_gray=False):
        if img is None:
            return
            
        if len(img.shape) == 2 or is_gray:
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], 
                         QImage.Format_Grayscale8)
        else:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], 
                         rgb_image.strides[0], QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), 
                                     Qt.KeepAspectRatio))

    def update_histogram(self, img):
        if img is None:
            return
            
        plt.clf()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        plt.hist(img.ravel(), 256, [0, 256])
        plt.title('Histogram')
        self.histogram_canvas.draw()

    def apply_sobel(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            self.processed_image = np.uint8(sobel)
            self.display_image(self.processed_image, self.processedLabel, True)
            self.update_histogram(self.processed_image)

    def apply_canny(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.processed_image = edges
            self.display_image(edges, self.processedLabel, True)
            self.update_histogram(edges)

    def apply_average_filter(self):
        if self.image is not None:
            filtered = cv2.blur(self.image, (5, 5))
            self.processed_image = filtered
            self.display_image(filtered, self.processedLabel)
            self.update_histogram(filtered)

    def apply_median_filter(self):
        if self.image is not None:
            filtered = cv2.medianBlur(self.image, 5)
            self.processed_image = filtered
            self.display_image(filtered, self.processedLabel)
            self.update_histogram(filtered)

    def apply_sharpening(self):
        if self.image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(self.image, -1, kernel)
            self.processed_image = sharpened
            self.display_image(sharpened, self.processedLabel)
            self.update_histogram(sharpened)

    def apply_blur(self):
        if self.image is not None:
            blurred = cv2.GaussianBlur(self.image, (5, 5), 0)
            self.processed_image = blurred
            self.display_image(blurred, self.processedLabel)
            self.update_histogram(blurred)

    def rotate_image(self, angle):
        if self.image is not None:
            height, width = self.image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(self.image, rotation_matrix, (width, height))
            self.processed_image = rotated
            self.display_image(rotated, self.processedLabel)
            self.update_histogram(rotated)

    def mirror_image(self, horizontal=True):
        if self.image is not None:
            if horizontal:
                mirrored = cv2.flip(self.image, 1)
            else:
                mirrored = cv2.flip(self.image, 0)
            self.processed_image = mirrored
            self.display_image(mirrored, self.processedLabel)
            self.update_histogram(mirrored)

    def apply_dilation(self):
        if self.processed_image is not None:
            # Check if the image is binary
            if len(np.unique(self.processed_image)) > 2:
                QMessageBox.warning(self, "Warning", "Please apply thresholding first!")
                return
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(self.processed_image, kernel, iterations=1)
            self.processed_image = dilated
            self.display_image(dilated, self.processedLabel, True)
            self.update_histogram(dilated)

    def apply_erosion(self):
        if self.processed_image is not None:
            # Check if the image is binary
            if len(np.unique(self.processed_image)) > 2:
                QMessageBox.warning(self, "Warning", "Please apply thresholding first!")
                return
            kernel = np.ones((3,3), np.uint8)
            eroded = cv2.erode(self.processed_image, kernel, iterations=1)
            self.processed_image = eroded
            self.display_image(eroded, self.processedLabel, True)
            self.update_histogram(eroded)

    def apply_skeletonization(self):
        if self.processed_image is not None:
            # Check if the image is binary
            if len(np.unique(self.processed_image)) > 2:
                QMessageBox.warning(self, "Warning", "Please apply thresholding first!")
                return
            
            # Convert to binary (0 and 1) for skeletonization
            # Ensure the image is properly thresholded
            binary = (self.processed_image > 127).astype(np.uint8)
            
            # Convert to boolean array for skeletonize function
            binary_bool = binary.astype(bool)
            
            # Apply skeletonization
            skeleton = skeletonize(binary_bool)
            
            # Convert back to uint8 image (0 and 255)
            self.processed_image = skeleton.astype(np.uint8) * 255
            
            # Display the result
            self.display_image(self.processed_image, self.processedLabel, True)
            self.update_histogram(self.processed_image)

    def compute_center_of_gravity(self):
      if self.processed_image is not None:
        # Check if the image is binary
        if len(np.unique(self.processed_image)) > 2:
            QMessageBox.warning(self, "Warning", "Please apply thresholding first!")
            return
        
        # Convert to binary (0 and 1) for centroid calculation
        binary = (self.processed_image > 127).astype(np.uint8)
        moments = cv2.moments(binary)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Convert grayscale to BGR to allow colored drawing
            color_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)

            # Draw red circle at center of gravity
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

            self.processed_image = color_image
            self.display_image(color_image, self.processedLabel)
            self.update_histogram(cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY))
        else:
            QMessageBox.warning(self, "Warning", "Could not compute center of gravity!")

    def apply_histogram_equalization(self):
        if self.image is not None:
            # Convert to grayscale if it's a color image
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image.copy()
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            self.processed_image = equalized
            self.display_image(equalized, self.processedLabel, True)
            self.update_histogram(equalized)

    def apply_contrast_stretching(self):
        if self.image is not None:
            # Convert to grayscale if it's a color image
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image.copy()
            
            # Find min and max pixel values
            min_val = np.min(gray)
            max_val = np.max(gray)
            
            # Apply contrast stretching
            stretched = ((gray - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
            self.processed_image = stretched
            self.display_image(stretched, self.processedLabel, True)
            self.update_histogram(stretched)

    def apply_manual_threshold(self):
        if self.image is not None:
            # Convert to grayscale if it's a color image
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image.copy()
            
            # Apply threshold
            threshold_value = self.thresholdSlider.value()
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            self.processed_image = binary
            self.display_image(binary, self.processedLabel, True)
            self.update_histogram(binary)

    def apply_otsu_threshold(self):
        if self.image is not None:
            # Convert to grayscale if it's a color image
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image.copy()
            
            # Apply Otsu's thresholding
            threshold_value = threshold_otsu(gray)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            self.processed_image = binary
            self.display_image(binary, self.processedLabel, True)
            self.update_histogram(binary)

    def apply_kapur_threshold(self):
        if self.image is not None:
            # Convert to grayscale if it's a color image
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image.copy()
            
            # Apply Kapur's thresholding
            threshold_value = threshold_yen(gray)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            self.processed_image = binary
            self.display_image(binary, self.processedLabel, True)
            self.update_histogram(binary)

    def save_image(self):
        if self.processed_image is not None:
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", 
                                                    "PNG Files (*.png);;JPEG Files (*.jpg)")
            if filePath:
                cv2.imwrite(filePath, self.processed_image)
                QMessageBox.information(self, "Saved", "Image saved successfully.")

    def no_image_warning(self):
        QMessageBox.warning(self, "Warning", "Please open an image first.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    ex.show()
    sys.exit(app.exec_())
