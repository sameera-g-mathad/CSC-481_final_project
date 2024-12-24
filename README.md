**Feature Descriptors for Image Classification**

This project explores the performance of three popular feature descriptors—Histogram of Oriented Gradients (HOG), Scale-Invariant Feature Transform (SIFT), and Oriented FAST and Rotated BRIEF (ORB)—for image classification tasks. The goal of this project is to analyze the strengths, limitations, and computational efficiency of each descriptor in distinguishing images across different categories. The implementation includes custom feature extraction and classification using Support Vector Machines (SVM).

*Project Overview*

In this project, we implemented the following feature descriptors from scratch:

* HOG: Focuses on detecting edges and gradients, ideal for images with structured patterns.

* SIFT: Designed for scale and rotation invariance, known for its robustness in feature matching.

* ORB: Combines FAST for keypoint detection with BRIEF descriptors, optimized for speed and efficiency.

The project evaluates each descriptor across three different datasets:
* Buildings vs. Forests: Focuses on structured images with distinct edges.

* Dogs vs. Cats: Tests how each descriptor handles images with complex textures.

* Airplanes, Cars, Ships: A multi-class dataset with distinct shapes and varying levels of complexity.

Due to computational limitations, we simplified the SIFT and ORB implementations by skipping the orientation assignment step, affecting their rotation invariance capabilities.

*Key Features*

* HOG: Effective for structured objects, achieving high accuracy on datasets with clear edges (98.9% on Buildings vs. Forests).

* SIFT: Flexible for scale and rotation, though accuracy is reduced due to simplifications in our implementation (54% on Dogs vs. Cats with rotated images).

* ORB: Optimized for speed and efficiency, but struggles with rotated images without proper orientation processing (52% on Dogs vs. Cats with rotated images).
