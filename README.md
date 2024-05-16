# Artwork Classification and Similarity through Low Information Learning

This project aims to classify artwork and measure similarity between artworks in a low information environment using deep learning techniques. The architectures used include Vision Transformers, Convolutional Neural Networks (CNNs), and Siamese Neural Networks.

## Introduction
The aim of this project is to leverage deep learning to create an objective method for identifying and measuring the similarity between artworks. Traditional methods of identifying artwork can be time-consuming and subjective. By using Vision Transformers, CNNs, and Siamese Neural Networks, we aim to improve the accuracy and efficiency of this process in a low information environment.

## Dataset
We used the [Best Artwork of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/data) dataset from Kaggle, which comprises a curated collection of artworks from the 50 most influential artists of all time. The number of paintings per artist ranges from 24 to 877.

## Technical Approach

### Data Pre-processing
We tested three datasets:
1. **Original**: From Kaggle, reduced to 20 artists.
2. **Oversampled**: Duplicated images to balance the number of images per class.
3. **Augmented**: Augmented images to increase dataset size and variability.

### Transfer Learning with CNNs
We used MobileNetV2, a convolutional neural network optimized for mobile and edge devices. The pre-trained MobileNetV2 model was fine-tuned to classify paintings based on artistic styles.

### Transfer Learning with Vision Transformers
Vision Transformers (ViTs) process images by splitting them into fixed-size patches. We used a pre-trained ViT and fine-tuned it to identify artistic styles from the paintings.

### Siamese Neural Networks
Siamese Neural Networks use a similarity function between input data to quantify similarity. We experimented with contrastive loss and triplet loss to train the Siamese networks for image classification and similarity measurement.

## Experiments and Results

### Transfer Learning Classification Experiments
We ran experiments on three data subsets (original, augmented, and oversampled) using Vision Transformers and CNNs. Here are some results:

- **Vision Transformer**: Best performance with the oversampled dataset.
- **CNN**: Best performance with the augmented dataset.

### Siamese Network Image Classification Experiment
We used the Siamese network with contrastive loss and evaluated it on the three data subsets. The oversampled dataset performed best.

### Image Similarity with Siamese Network Experiment
Using the triplet loss function, we trained the model to measure similarity between artworks. The model achieved a validation accuracy of 0.61 after 20 epochs.

## Conclusion
We achieved moderate success in classifying and measuring similarity between artworks in a low information environment. The Siamese network performed best, indicating its effectiveness in learning pairwise similarities.
