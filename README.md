🌲 Dead Tree Segmentation Pipeline 🍂
This project provides an automated pipeline for detecting and segmenting dead trees from aerial imagery using RGB and NRG data.
The pipeline performs preprocessing, vegetation analysis, and segmentation based on configurable thresholds.
🚀 Installation & Usage
1. Clone the repository

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
