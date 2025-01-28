# AI Programming with Python Nanodegree - Image Classifier

## Overview
This repository contains the code and documentation for the AI Programming with Python Nanodegree project. The project involves developing an image classifier using PyTorch, and converting it into a command-line application.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Examples](#examples)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-classifier.git
   cd image-classifier

2. Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
train.py: Script to train the image classifier model.

predict.py: Script to make predictions using the trained model.

model.pth: Trained model file.

cat_to_name.json: Mapping of category labels to category names.

requirements.txt: List of required dependencies.

README.md: This documentation file.

## Usage
Training the Model
To train the model, run the following command:

```bash
python train.py --data_dir /path/to/data --save_dir /path/to/save --arch
```

Predicting with the Model
To make predictions using the trained model, run:

```bash
python predict.py --image_path /path/to/image --checkpoint /path/to/save/model.pth --top_k 5 --category_names cat_to_name.json --gpu
```
## Examples
Below are some examples of how to use the command-line application:

Training the model:

```bash
python train.py --data_dir flowers --save_dir checkpoints --arch alexnet --learning_rate 0.001 --hidden_units 256 --epochs 10 --gpu
```
Predicting an image:

```bash
python predict.py --image_path flowers/test/1/image_06743.jpg --checkpoint checkpoints/checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu


