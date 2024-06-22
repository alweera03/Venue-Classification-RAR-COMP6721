============================================================
Venue Classification Project - COMP6721
============================================================

Welcome to the Venue Classification Project repository. This project focuses on classifying images into one of five venue categories using a variety of machine learning techniques, including supervised and semi-supervised decision tree classifiers, as well as convolutional neural networks (CNNs).

Table of Contents:
------------------
1. Introduction
2. Datasets
3. Models
   3.1 Supervised Decision Tree Classifier
   3.2 Semi-Supervised Decision Tree Classifier
   3.3 Convolutional Neural Networks (CNNs)
       3.3.1 Model 1
       3.3.2 Model 2
       3.3.3 Model 3
4. IPython Notebooks
5. Installation
6. Pre-trained Models
7. Usage
  

## Introduction
This project implements machine learning models to classify images into five venue categories: Airport, Beach, Restaurant, Library, and Gym. It includes both traditional supervised learning and semi-supervised learning techniques for decision tree classifiers, as well as deep learning with CNNs.
This project focuses on classifying images into one of five venue categories: Airport, Beach, Restaurant, Library, and Gym. We have trained both supervised and semi-supervised decision tree classifiers to achieve this goal.


## Datasets
The datasets used for this project can be accessed from the following links:
- [Airport Dataset](https://images.cv/dataset/airport-inside-image-classification-dataset)
- [Beach Dataset](https://images.cv/dataset/beach-image-classification-dataset)
- [Restaurant Dataset](https://images.cv/dataset/restaurant-image-classification-dataset)
- [Gym Dataset 1](https://images.cv/dataset/gym-image-classification-dataset)
- [Gym Dataset 2](https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images)
- [Library Dataset](https://images.cv/dataset/library-image-classification-dataset)

## Models
Both the supervised and semi-supervised models leverage feature extraction techniques for image classification:

### Preprocessing Steps

1. **Image Loading and Feature Extraction**:
   - Images are loaded from the specified directories (`train_folder`, `val_folder`, `test_folder`) using OpenCV (`cv2`).
   - Each image is resized to a predefined size (`IMG_HEIGHT x IMG_WIDTH`).

2. **Feature Extraction Functions**:
   - **Color Histogram Extraction**: 
     - Converts images to HSV color space and computes a normalized color histogram using `cv2.calcHist()`.
   - **Grayscale Pixel Extraction**: 
     - Converts images to grayscale and flattens the pixel values.

3. **Combining Features**:
   - Extracted color histograms and grayscale pixel values are concatenated (`np.hstack()`) to form combined feature vectors for each image.

4. **Normalization**:
   - Features are standardized using `StandardScaler()` from `sklearn.preprocessing`.
   - Standardization ensures that all features have a mean of 0 and a standard deviation of 1, which helps in improving convergence and performance of the models.

5. **Label Encoding**:
   - Labels (venue categories) are encoded using `LabelEncoder()` from `sklearn.preprocessing`.
   - Encoded labels (`y_train_enc`, `y_val_enc`, `y_test_enc`) are used for model training and evaluation.

### Supervised Decision Tree Classifier
The supervised model is trained using labelled data from the above-mentioned datasets. This model learns to classify images based on the features present in the labelled examples.

The supervised learning approach involves training a Decision Tree Classifier using predefined hyperparameters:

1. **We defined different Hyperparameters for Set 1 and Set 2**:
   - Hyperparameters (`params_set_1`) are predefined to configure the Decision Tree Classifier:
     - `criterion`: Split criterion for decision tree ('gini' or 'entropy').
     - `max_depth`: Maximum depth of the tree.
     - `min_samples_split`: Minimum number of samples required to split an internal node.
     - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
     - `max_features`: Number of features to consider when looking for the best split ('sqrt' for square root of the total number of features).


### Semi-Supervised Decision Tree Classifier
The semi-supervised model leverages both labelled and unlabeled data to improve classification performance. It initially trains on the labelled data and then refines its understanding using the unlabeled data.
### Code Explanation

1. **Splitting the Dataset**:
   - The training set (`X_train_flat`, `y_train_enc`) is split into a labeled subset (`X_train_labeled`, `y_train_labeled`) and an unlabeled subset (`X_train_unlabeled`).

2. **Semi-Supervised Learning Function** (`semi_supervised_learning`):
   - **Initialization**: Start with a Decision Tree Classifier (`clf`) without any labeled samples.
   - **Iteration**: For a predefined number of iterations or until certain conditions are met:
     - Train the classifier (`clf`) on the current labeled set (`X_labeled`, `y_labeled`).
     - Predict probabilities for the unlabeled samples (`X_unlabeled`).
     - Identify high-confidence predictions (`confidence_threshold`) and add them to the labeled set.
     - Remove these samples from the unlabeled set to avoid re-labeling.
     - Repeat until no more high-confidence samples are identified or the unlabeled set is exhausted.

3. **Evaluate the Model**:
   - After semi-supervised learning, evaluate the trained classifier (`clf_semi_supervised`) on the test set (`X_test_flat`, `y_test_enc`).
   - Compute and print performance metrics such as accuracy, precision, recall, and F1-score.
   - Generate a confusion matrix to visualize the model's performance across different venue categories.

### Training Process
1. **Data Preprocessing**: Images are resized and normalized to ensure consistency.
2. **Feature Extraction**: Features are extracted from the images using colour histogram and the grayscale pixels of the images
3. **Model Training**: The supervised and semi-supervised decision tree classifiers are trained on the extracted features with different hyperparameters.
4. **Evaluation**: The models are evaluated on a separate validation set using metrics such as accuracy, precision, recall, and F1-score.

### Convolutional Neural Network (CNN)
The CNN model is trained to classify images using deep learning techniques. The preprocessing steps for the CNN are as follows:

**Training Data Preprocessing**:
1. **Convert to Grayscale**: All images are converted to grayscale.
2. **Resize Images**: Images are resized to a fixed size of 128x128 pixels.
3. **Random Horizontal Flip**: Images are randomly flipped horizontally to augment the dataset.
4. **Random Rotation**: Images are randomly rotated by up to 10 degrees.
5. **Random Resized Crop**: A random crop of the image is taken and resized back to 128x128 pixels, with the crop size varying between 80% and 100% of the original image size.
6. **Convert to Tensor**: The images are converted to PyTorch tensors.
7. **Normalize**: The pixel values of the images are normalized to have a mean of 0.5 and a standard deviation of 0.5.

**Validation and Test Data Preprocessing**:
1. **Convert to Grayscale**: All images are converted to grayscale.
2. **Resize Images**: Images are resized to a fixed size of 128x128 pixels.
3. **Convert to Tensor**: The images are converted to PyTorch tensors.
4. **Normalize**: The pixel values of the images are normalized to have a mean of 0.5 and a standard deviation of 0.5.

### Training Process 
1. **Data Preprocessing**: Images are resized and normalized to ensure consistency.
2. **Feature Extraction**: Features are extracted from the images using a colour histogram and the grayscale pixels of the images
3. **Model Training**: The supervised and semi-supervised decision tree classifiers are trained on the extracted features with different hyperparameters.
4. **Evaluation**: The models are evaluated on a separate validation set using accuracy, precision, recall, and F1-score metrics.

### Convolutional Neural Network-Model 1 (CNN)
The CNN model architecture consists of the following layers:

1. **Convolutional Layers**:
   - **Conv1**: 32 filters, each of size 3x3, with a stride of 1 and padding of 1. This layer processes grayscale input images (1 channel).
   - **Max Pooling**: 2x2 pooling with a stride of 2, reducing the spatial dimensions by half after each pooling operation.
   - **Conv2**: 64 filters, each of size 3x3, with a stride of 1 and padding of 1.
   - **Max Pooling**: Another 2x2 pooling with a stride of 2.

2. **Fully Connected Layers**:
   - **FC1**: Fully connected layer with 512 neurons, taking the flattened output from the last convolutional layer (64 * 32 * 32).
   - **FC2**: Fully connected layer with 5 neurons, corresponding to the number of output classes (assuming 5 venue categories).

3. **Activation Functions**:
   - **ReLU**: Used after each convolutional layer and the first fully connected layer to introduce non-linearity.

4. **Loss Function and Optimizer**:
   - **Loss Function**: CrossEntropyLoss, suitable for multi-class classification tasks.
   - **Optimizer**: Adam optimizer with a learning rate of 0.001, used to update the model parameters based on computed gradients.
  
### Modified Convolutional Neural Network (CNN Model 2)
The modified CNN model architecture (`VenueCNN_Modified`) is designed as follows:

1. **Convolutional Layers**:
   - **Conv1**: 32 filters of size 3x3, with a stride of 1 and padding of 1, processing grayscale input images (1 channel).
   - **Max Pooling**: 2x2 pooling with a stride of 2.
   - **Conv2**: 64 filters of size 3x3, with a stride of 1 and padding of 1.
   - **Max Pooling**: 2x2 pooling with a stride of 2.
   - **Conv3**: 128 filters of size 3x3, with a stride of 1 and padding of 1. Added an extra convolutional layer.
   - **Max Pooling**: 2x2 pooling with a stride of 2.

2. **Fully Connected Layers**:
   - **FC1**: Fully connected layer with 512 neurons, adjusted for the additional conv layer.
   - **FC2**: Fully connected layer with 256 neurons, added for deeper representation learning.
   - **FC3**: Output layer with 5 neurons, corresponding to the number of venue categories.

3. **Activation Functions**:
   - **ReLU**: Used after each convolutional and fully connected layer to introduce non-linearity.

4. **Loss Function and Optimizer**:
   - **Loss Function**: CrossEntropyLoss, suitable for multi-class classification tasks.
   - **Optimizer**: Adam optimizer with a learning rate of 0.001, used to update the model parameters based on computed gradients.
  
### Further Modified Convolutional Neural Network (CNN Model 3)
The `VenueCNN_Modified` model architecture has been enhanced with additional layers and batch normalization:

1. **Convolutional Layers**:
   - **Conv1**: 32 filters of size 3x3, with a stride of 1 and padding of 1, followed by batch normalization.
   - **Max Pooling**: 2x2 pooling with a stride of 2.
   - **Conv2**: 64 filters of size 3x3, with a stride of 1 and padding of 1, followed by batch normalization.
   - **Max Pooling**: 2x2 pooling with a stride of 2.
   - **Conv3**: 128 filters of size 3x3, with a stride of 1 and padding of 1, followed by batch normalization.
   - **Max Pooling**: 2x2 pooling with a stride of 2.
   - **Conv4**: 256 filters of size 3x3, with a stride of 1 and padding of 1, added for deeper representation learning, followed by batch normalization.
   - **Max Pooling**: 2x2 pooling with a stride of 2.

2. **Dropout Layer**:
   - Applied after the first fully connected layer (`fc1`) with a dropout rate of 0.5 to prevent overfitting.

3. **Fully Connected Layers**:
   - **FC1**: Fully connected layer with 512 neurons, adjusted for the additional convolutional layers and pooling.
   - **FC2**: A fully connected layer with 256 neurons was added for deeper representation learning.
   - **FC3**: Output layer with 5 neurons, corresponding to the number of venue categories.

4. **Activation Functions**:
   - **ReLU**: Used after each convolutional and fully connected layer to introduce non-linearity.      

### Training Process
1. **Data Preprocessing**: Images are resized, normalized, and augmented (random horizontal flip, rotation, and crop) to enhance model generalization.
2. **Model Training**: The CNN model is trained on the training dataset over 25 epochs using the defined loss function and optimizer. Training progress and validation metrics (loss, accuracy) are printed after each epoch.
3. **Evaluation**: After training, the model is evaluated on a separate test dataset to assess its performance using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize the classification results.
  

- [Download Model 1](https://drive.google.com/uc?export=download&id=1-1dG-cW7nnl-wyuyb3-jKk2N1Ihhuerk)
- [Download Model 2](https://drive.google.com/uc?export=download&id=1d9aiEwJc1kx4UgBc9yActoMbBZviI8VS)
- [Download Model 3](https://drive.google.com/uc?export=download&id=1_Is_iL0SJOmfOFYj7BR3r6pu3Yc614l1)


## IPython Notebooks
- [Supervised and Semi-Supervised Decision Tree Classification](Supervised%20and%20Semi-Supervised%20Decision%20Tree%20Classification.ipynb): Detailed notebook covering training, validation, and comparison of decision tree classifiers.
- [Venue Classification using CNN](COMP6721_Venue_Classification_using_CNN.ipynb): Notebook focusing on CNN implementation for venue classification, including training and evaluation.

## Installation
To run the project, ensure you have the following dependencies installed:
- `numpy`
- `opencv-python`
- `scikit-learn`
- `torch` (for PyTorch models)
- `torchvision` (for PyTorch models)


## Pre-trained Models
- [Download Model 1](https://drive.google.com/uc?export=download&id=1-1dG-cW7nnl-wyuyb3-jKk2N1Ihhuerk)
- [Download Model 2](https://drive.google.com/uc?export=download&id=1d9aiEwJc1kx4UgBc9yActoMbBZviI8VS)
- [Download Model 3](https://drive.google.com/uc?export=download&id=1_Is_iL0SJOmfOFYj7BR3r6pu3Yc614l1)
## Usage
1. Download the models using the links provided above.
2. Load the desired model in your project.
3. Preprocess input images according to the requirements of the chosen model.
4. Use the model to obtain predictions on your data.
