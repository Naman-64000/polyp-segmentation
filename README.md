# Polyp Segmentation Project

## Overview
This project is focused on **Polyp Segmentation** using deep learning models, particularly the **TransUNet** architecture. It involves segmenting polyps from colonoscopy images for early detection and diagnosis of colorectal cancer.

### Key Features:
- **Polyp segmentation** from colonoscopy images.
- Utilizes **TransUNet** architecture for segmentation tasks.
- Trains the model on publicly available colonoscopy datasets (e.g., Kvasir-SEG).
- Provides scripts for training and evaluating the model.

## Requirements
Make sure you have the following dependencies installed to run the project:
- python-utils 3.9.1
- TensorFlow 2.18.0
- numpy 1.26.4
- opencv-python 4.10.0.84
- matplotlib 3.10.0
- pandas 2.2.2

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## Installation Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-username/polyp-segmentation.git
cd polyp-segmentation
```

2. Set up a virtual environment (optional but recommended):
```bash 
python3 -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

3. Install the required dependencies:
```bash 
pip install -r requirements.txt
```
## Usage

To train the model, run the following command:

```bash 
python train.py --epochs 50 --batch_size 8 --learning_rate 0.001
```




To evaluate the model on a test dataset:
```bash
python evaluate.py --model_path models/transunet_model.h5 --test_data data/test_images
```

To segment a new image:
```bash 
python predict.py --model_path models/transunet_model.h5 --image_path images/new_image.jpg
```
## Data
This project uses the **Kvasir-SEG dataset**, which contains labeled colonoscopy images for training and testing. You can download the dataset from [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/).


## Data Preprocessing:
- Images are resized to a consistent input size.
- Normalization is performed to scale pixel values between 0 and 1.
- Data augmentation techniques (like rotation and flipping) are applied to improve model robustness.

## Model Architecture

This project uses the **TransUNet** architecture, which combines the strengths of **Transformers** and **UNet** for image segmentation tasks. The transformer component helps the model capture long-range dependencies in the image, while the UNet architecture ensures precise pixel-level segmentation.

## Key Components:

- **Encoder**: A series of convolutional layers and transformers to extract feature maps.
- **Decoder**: Uses upsampling and skip connections to generate pixel-level segmentation masks.

## Results

Here are some example results of the segmentation:

- **Input Image with Predicted Segmentation**:  
  ![Input Image with Predicted Segmentation](https://github.com/Naman-64000/polyp-segmentation/blob/main/result.png?raw=true)



### Evaluation Metrics:

- **Dice Coefficient**: 0.726
- **IoU (Intersection over Union)**: 0.652

These metrics show that the model performs well in accurately segmenting polyps from colonoscopy images, with a high Dice coefficient indicating a strong overlap between the predicted and ground truth masks.


### Code of Conduct

Please follow the [Code of Conduct](./CODE_OF_CONDUCT.md) when contributing to this project. 
