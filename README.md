Dead Tree Segmentation Pipeline 🌲🍂
This project provides an automated pipeline for detecting and segmenting dead trees from aerial imagery using RGB and NRG data.

🚀 Installation & Usage
To get started, follow these steps to set up your environment and run the pipeline:

1. Clone the repository

```Bash
git clone https://github.com/YourUsername/dead-tree-segmentation.git
cd dead-tree-segmentation
```
2. Install dependencies

It is recommended to use a virtual environment. Install all required libraries using:

```Bash
pip install -r requirements.txt
```

3. Run the pipeline

Make sure your data is placed in the test data folder, then execute the main script:

```Bash
python main.py
```

4. Optional: Override thresholds

You can manually adjust the Hue thresholds via command line arguments:

```Bash
python main.py -h_min 0.72 -h_max 0.92
```

⚙️ Configuration
The config.yaml file allows you to manage paths and segmentation thresholds.

YAML
paths:
  rgb: "./test data/RGB_images/*.png"
  nrg: "./test data/NRG_images/*.png"
  masks: "./test data/masks/*.png"
