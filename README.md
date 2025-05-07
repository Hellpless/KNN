# KNN (English version)
 Weighted K-Nearest Neighbors (KNN) Classifier with GUI

This project implements a K-Nearest Neighbors (KNN) classifier with a specific weighting scheme for dimensions and neighbor distances. It includes a graphical user interface (GUI) built with Tkinter for interactive data input, classification, and visualization of 2D/3D data points.

This project was developed Schritt für Schritt (step-by-step) as a learning exercise.

## Features

* **Weighted KNN Algorithm**:
    * Calculates dimension weights based on the formula: `W_ed = (mean of squares of values in dimension d)^(-1)`.
    * Supports multiple distance metrics for finding neighbors:
        * Weighted Euclidean Distance
        * Weighted Manhattan Distance
        * Weighted Chebyshev Distance
    * Classifies new data points based on the weighted votes of its 'k' nearest neighbors (weight = 1/distance).
* **Graphical User Interface (GUI)**:
    * Built using Python's Tkinter library.
    * Allows users to input training data (features and class labels).
    * Supports two classification modes:
        1.  **New Point Classification**: Classify an externally provided data point.
        2.  **Leave-One-Out (LOO) Classification**: Classify an existing point from the training set by temporarily excluding it.
    * Input fields for the number of neighbors (`k`) and selection of the distance metric.
    * Displays detailed classification results, including the predicted class, total weights for each class, and details of the k-nearest neighbors.
    * **2D/3D Data Visualization**:
        * Uses Matplotlib to plot training data, the point to be classified, and its nearest neighbors.
        * Supports visualization for datasets with 2 or 3 features.
        * If Matplotlib is not available, the GUI will still function without the graph.
* **Data Generation**:
    * Includes a script (`1000.py` or `knn_dummy_data_1000.txt`) to generate dummy datasets for testing.

## File Structure and Evolution

The project evolved through several Python script versions, demonstrating the progressive development of features:

* `KNN_Understandable.py`: Likely an initial, more commented, or simplified version of the KNN logic for understanding.
* `KNN.py`: Core KNN logic, possibly a command-line version or the foundational algorithm.
* `KNN_GUI_V0.1.py`: First version of the GUI, basic data input, and classification.
* `KNN_GUI_V0.2(Graph).py`: Added 2D graph visualization to the GUI.
* `KNN_GUI_V0.3(Graph_3D).py`: Extended graph visualization to support 3D data.
* `KNN_GUI_V0.4(FromMyOwn).py`: The latest version, incorporating Leave-One-Out (LOO) classification mode and potentially other refinements based on the user's ("FromMyOwn") specific requirements or school assignments. This is likely the main application file to run.
* `1000.py`: Python script to generate 1000 dummy data points (as seen in `knn_dummy_data_1000.txt`).
* `knn_dummy_data_1000.txt`: A text file containing 1000 generated dummy data samples for testing.
* `requirements.txt`: Lists project dependencies (e.g., `matplotlib`).

**The main application to run is likely `KNN_GUI_V0.4(FromMyOwn).py`.**

## How to Run

1.  **Prerequisites**:
    * Python 3.x
    * Tkinter (usually included with Python)
    * Matplotlib (for graph visualization). If not installed, run:
        ```bash
        pip install matplotlib
        ```
    * Check `requirements.txt` for any other specific versions if provided.

2.  **Clone the repository (if applicable) or download the files.**

3.  **Navigate to the `KNN` directory in your terminal.**

4.  **Run the main application**:
    ```bash
    python KNN_GUI_V0.4(FromMyOwn).py
    ```

5.  **Using the GUI**:
    * Input training data in the format `feature1,feature2,[feature3],class_label` (one point per line).
    * Choose the classification mode:
        * **New Point**: Enter features for the new point to classify.
        * **Existing Point (LOO)**: Click "Aktualizuj výber bodov pre LOO" to populate the dropdown, then select a point.
    * Enter the value for `k` (number of neighbors).
    * Select the desired distance metric.
    * Click "Klasifikuj".
    * Results will be displayed in the text area, and a graph will be shown if Matplotlib is available and data is 2D or 3D.

## Based on School Assignment

This project appears to be based on a school assignment, incorporating specific formulas for dimension weighting and distance calculations as discussed and potentially demonstrated in class examples. The "Leave-One-Out" (LOO) classification mode was added to align with exercises where existing data points are classified by their peers.

---
