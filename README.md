# Housing Price Prediction (University Group Project)

This repository contains the code and the **full technical report** for a university assignment on Pattern Recognition. The goal was to predict housing prices using the  Housing dataset, comparing statistical methods with machine learning approaches.

**[Click here to read the Full Project Report (PDF)](./pattern_recognition.pdf)**

## Project Context
* **Type:** Group Assignment (2 Members)
* **Course:** Pattern Recognition

## Approaches Implemented
We implemented four distinct strategies to showcase different levels of complexity:

1.  **Data Visualization (`Data_visualization.py`):**
    * Data Analysis using Matplotlib and Seaborn.
2.  **Mathematical Approach (`Least_squares.py`):**
    * Implementation of the **Least Squares** method from scratch using NumPy.
3.  **Algorithmic Approach (`Linear_perceptron.py`):**
    * Implementation of a **Linear Perceptron** from scratch.
4.  **Deep Learning Approach (`Multi_layer_neural_network.py`):**
    * A Multi-Layer Perceptron (MLP) built with **PyTorch**.
    * Uses 10-Fold Cross-Validation.

## How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Analysis:**
    ```bash
    python Multi_layer_neural_network.py
    ```
