# Image Classifier for Udacity AI Programming with Python

This project is part of the Udacity AI Programming with Python Nanodegree with AWS scholarship. It's an image classifier that allows users to train a deep learning model on a dataset of images and make predictions on new images using a trained model. The project consists of three main components: an HTML file (Jupyter Notebook), and two Python scripts for training and prediction for command line application.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Instructions](#instructions)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Overview

This project is a part of the Udacity AI Programming with Python Nanodegree and focuses on image classification using deep learning. It allows you to train a deep neural network to classify images into predefined categories. The project demonstrates your knowledge and skills in working with deep learning models, data preprocessing, and model evaluation.

## Features

- **Image Classification:** Train a deep learning model to classify images into categories.
- **Visual Interface:** Use the provided HTML file to interact with the project.
- **Command Line Interface:** Utilize the command-line Python scripts to train and predict.

## Instructions

The project includes two main Python scripts created as command line applications: `train.py` for training the model and `predict.py` for making predictions. Here are detailed instructions for using the project:

### Command Line Application

1. Run the `train.py` script with the following command:

```bash
!python train.py path_to_data --save_dir checkpoints/ --arch vgg16 --learning_rate 0.001 --gpu
```

2. Run the `predict.py` script with the following command:

```bash
!python predict.py path_to_image path_to_checkpoint --top_k 3 --category_names cat_to_name.json --gpu
```

## Project Structure

The project includes the following components:

- Jupyter Notebook: Included in HTML format.
- train.py: The Python script for training the image classifier model.
- predict.py: The Python script for making predictions on new images.
- cat_to_name.json: A JSON file mapping category names to class labels.
- README.md: This file provides an overview of the project.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL (Pillow)
- Jupyter Notebook (for interactive usage)

You can install these dependencies using pip or conda as appropriate for your environment.



