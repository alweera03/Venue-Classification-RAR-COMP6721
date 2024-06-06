# Venue Classification Project

This project focuses on classifying images into one of five venue categories: Airport, Beach, Restaurant, Library, and Gym. We have trained both supervised and semi-supervised decision tree classifiers to achieve this goal.

## Table of Contents
- [Datasets](#datasets)
- [Models](#models)

## Datasets
The datasets used for this project can be accessed from the following links:
- [Airport Dataset](https://images.cv/dataset/airport-inside-image-classification-dataset)
- [Beach Dataset](https://images.cv/dataset/beach-image-classification-dataset)
- [Restaurant Dataset](https://images.cv/dataset/restaurant-image-classification-dataset)
- [Gym Dataset 1](https://images.cv/dataset/gym-image-classification-dataset)
- [Gym Dataset 2](https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images)
- [Library Dataset](https://images.cv/dataset/library-image-classification-dataset)

## Models
### Supervised Decision Tree Classifier
The supervised model is trained using labelled data from the above-mentioned datasets. This model learns to classify images based on the features present in the labelled examples.

### Semi-Supervised Decision Tree Classifier
The semi-supervised model leverages both labelled and unlabeled data to improve classification performance. It initially trains on the labelled data and then refines its understanding using the unlabeled data.

### Training Process
1. **Data Preprocessing**: Images are resized and normalized to ensure consistency.
2. **Feature Extraction**: Features are extracted from the images using color histogram and the grayscale pixels of the images
3. **Model Training**: The decision tree classifiers are trained on the extracted features.
4. **Evaluation**: The models are evaluated on a separate validation set using metrics such as accuracy, precision, recall, and F1-score.

### IPython Notebook
For detailed walkthroughs and explanations of the training process and results, refer to the IPython Notebook file [COMP6721_Venue_Classification code.ipynb](COMP6721_Venue_Classification%20code.ipynb).
