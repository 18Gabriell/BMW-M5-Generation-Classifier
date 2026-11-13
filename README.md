# BMW M5 Generation Classifier

## Project Description

A Python machine learning project that classifies BMW M5 images into seven generations using transfer learning with MobileNetV2.

## Technologies
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- MobileNetV2

## Supported Generations
E28, E34, E39, E60, F10, F90, G90.

## Model Training
- Image resolution: 224 × 224 pixels
- Batch size: 8
- Training epochs: 25
- Training split: 75%
- Test split: 25%
- Data augmentation: rotation, zoom, translation and horizontal flipping

## Results
- Test accuracy: **82.86%**
- Correct predictions: **29 out of 35 test images**

The evaluation uses a relatively small dataset, so the results are preliminary.

## Future Improvements
- Expand the image dataset
- Improve model generalization
- Add a separate validation dataset
- Create an interface for predicting BMW M5 generations from uploaded images