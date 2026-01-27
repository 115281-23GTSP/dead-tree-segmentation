Key Features

Multispectral Fusion: Combines RGB and NRG (Infrared) data for superior detection accuracy.

NDVI Analysis: Automatically differentiates dead biomass from healthy vegetation.

Dual Thresholding: Processes images via HSV color space and vegetation indices simultaneously.

Evaluation Suite: Calculates IoU, Confusion Matrix, and F1-Score metrics automatically.



Segmentation Pipeline

Preprocessing: Standardizes images to 256x256 using nearest-neighbor interpolation.

Spectral Indexing: Computes NDVI to isolate biological signatures of dead wood.

https://eos.com/blog/ndvi-faq-all-you-need-to-know-about-ndvi/<img width="580" height="305" alt="image" src="https://github.com/user-attachments/assets/e8faac8d-e667-43df-9fe1-1ea21e84c93a" />


Color Masking: Filters specific grey and pink hues in the HSV color space.

Post-Processing: Merges masks via logical AND followed by hole filling and dilation.

Validation: Compares the final mask against ground truth to generate performance charts.
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

Visual Explanation:
<img width="1434" height="405" alt="Zrzut ekranu 2026-01-12 o 22 34 59" src="https://github.com/user-attachments/assets/02ab007e-2de2-44fa-a339-fb4c4f0ab881" />

<img width="785" height="472" alt="image" src="https://github.com/user-attachments/assets/a00df9c7-9c43-4656-874d-f11e5fb3a546" />


hot to make venv

```bash
go to your folder
```

then

```bash
python3 -m venv .venv
```

```bash
source venv/bin/activate
```

and athe end of the venv add requirements.txt

```bash
pip install -r requirements.txt
```
