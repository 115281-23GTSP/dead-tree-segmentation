Here is the professional description of the Key Features and Segmentation Pipeline for your README.md file in English:

🚀 Key Features
Multi-Spectral Data Integration: Leverages the synergy between RGB and NRG (Near-Infrared, Red, Green) imagery to enhance detection accuracy.

Automated NDVI Analysis: Automatically calculates the NDVI (Normalized Difference Vegetation Index) to distinguish dead biomass from healthy living vegetation.

Advanced Image Refinement: Utilizes morphological operations, including binary hole filling and dilation (using a disk element of radius 5), to create continuous and robust segmentation masks.

Dual-Thresholding Strategy: Implements a two-pronged approach for segmentation:

HSV Color Space Thresholding: Specifically targets the pinkish/greyish hues characteristic of dead wood.

Vegetation Index Thresholding: Filters pixels based on biological spectral signatures.

Comprehensive Evaluation Suite: Includes a complete analytical module to calculate:

IoU (Intersection over Union): To measure the overlap accuracy between predicted and ground truth masks.

Confusion Matrix: Visualized via heatmaps to assess Accuracy, Precision, Recall, and F1-Score.

Flexible CLI & Configuration: Manage paths, sizes, and thresholds through a config.yaml file or via command-line arguments (CLI) for rapid testing.

🛠 Segmentation Pipeline
The data processing workflow follows a structured sequence to ensure high-quality results:

1. Data Loading & Standardization

The system loads RGB images, NRG images, and ground truth masks.

Images are resized to the target dimensions (e.g., 256x256) using order 0 interpolation (nearest-neighbor) to maintain the integrity of binary masks and categorical data.

2. Spectral Analysis (NDVI Calculation)

The algorithm extracts the Near-Infrared (NIR) and Red channels from the NRG images to compute the NDVI:

NDVI= 
NIR+Red+10 
−10
 
NIR−Red
​	
 
Pixels falling within the range of 0.2 to 0.4 are classified as potential dead wood areas.

3. HSV Color Segmentation

Simultaneously, the RGB image is converted to the HSV (Hue, Saturation, Value) color space. The pipeline filters pixels based on the Hue component to isolate the specific "dead wood" color palette defined in the configuration.

4. Mask Combination & Morphological Post-Processing

Logical Conjunction: The masks from both the NDVI and HSV stages are combined using a logical AND operation to minimize False Positives.

Hole Filling: Small gaps within detected objects are removed using binary_fill_holes.

Dilation: A dilation operation is applied to smooth edges and connect closely situated detected fragments into a single object.

5. Validation & Visualization

The final mask is compared against the ground truth mask to compute statistical metrics.

The pipeline generates bar charts for IoU, heatmaps for the confusion matrix, and a 6-panel visual comparison showing every stage of the process.

```bash
git clone https://github.com/YourUsername/dead-tree-segmentation.git
cd dead-tree-segmentation
```

3. Install dependencies
It is strongly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

5. Prepare the data

Place your input data in the following structure (default):
```bash
test data/
├── RGB_images/
├── NRG_images/
└── masks/
```
File paths can be modified in config.yaml.

6. Run the pipeline

```bash
python main.py
```

8. Optional: Override segmentation thresholds
You can manually adjust Hue thresholds via command-line arguments:

```bash
python main.py --h_min 0.72 --h_max 0.92
```

⚙️ Configuration
The config.yaml file allows you to manage input/output paths and segmentation parameters.
Example:

```bash
paths:
  rgb: "./test data/RGB_images/*.png"
  nrg: "./test data/NRG_images/*.png"
  masks: "./test data/masks/*.png"
```

🔄 Running the Project Directly from Git
If you want to run or update the project directly from GitHub, follow these steps:
First-time setup

```bash
git clone https://github.com/YourUsername/dead-tree-segmentation.git
cd dead-tree-segmentation
pip install -r requirements.txt
python main.py
```

Updating the project
When new changes are available in the repository:

```bash
git pull origin main
```

Then rerun:

```bash
python main.py
```

🛠 Requirements
Python 3.8+
Dependencies listed in requirements.txt
📌 Notes
Ensure RGB and NRG images are correctly aligned.
Threshold values may require tuning depending on lighting conditions and vegetation type.
