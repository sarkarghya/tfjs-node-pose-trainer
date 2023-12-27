# TensorFlow.js Pose Classification

This project uses TensorFlow.js to create an image classification model to recognize 5 different yoga poses - downdog, goddess, plank, tree, and warrior2.

## Overview

The project has the following structure:

- `data.js` - Loads images from the filesystem and processes them into tensors 
- `model.js` - Defines and trains a CNN model 
- `main.js` - Entry point that loads data and trains the model
- Other helpers and config files

The model is a convolutional neural network (CNN) with the following layers:

- Convolutional 
- Max pooling
- Dropout
- Flatten
- Dense

It is trained for 50 epochs with categorical crossentropy loss and adam optimizer.

## Usage

Check compatibility:
- I have verified usability on MacOS (M2,2022 chip) and on Ubuntu (Linux-ublts22043).
- Known issues include incompatibilty of latest tfjs node versions with Windows.

To train the model:

1. Clone the repo
2. Install dependencies with `npm install`
3. Make sure you have training data images in `./DATASET/TRAIN` and test data in `./DATASET/TEST`
4. Run `node main.js` (OR) `npm run train` to execute `main.js` and train the model

The trained model and its weights are saved in the repo itself. 

## Development

Please commit to ``dev-primary`` branch to suggest any changes.

Some ways the project can be improved:

- Add data augmentation 
- Use transfer learning from a pretrained model
- Deploy the model to make predictions on new images

## Author

Email: arghya[at]nyu[dot]edu