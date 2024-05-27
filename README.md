# Handwritten Digit Classification

This project demonstrates the process of building and training a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset.

## Project Structure

- `data/`: Contains instructions to download the MNIST dataset.
- `notebooks/`: Contains Jupyter notebook with the code and explanations.
- `src/`: Contains Python scripts for different stages of the project.
  - `data_preprocessing.py`: Script for data loading and preprocessing.
  - `model.py`: Script for building the CNN model.
  - `train.py`: Script for training the model.
  - `evaluate.py`: Script for evaluating the model.
  - `plot_results.py`: Script for plotting the training and validation results.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `requirements.txt`: Lists the dependencies required to run the project.

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Handwritten-Digit-Classification.git
    cd Handwritten-Digit-Classification
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

1. **Data Preprocessing**: 
    ```bash
    import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

    ```

2. **Train the Model**:
    ```bash
    python src/train.py
    ```

3. **Evaluate the Model**:
    ```bash
    python src/evaluate.py
    ```

4. **Plot Training and Validation Results**:
    ```bash
    python src/plot_results.py
    ```

### Usage

To run the Jupyter notebook:
```bash
jupyter notebook notebooks/MNIST_CNN.ipynb
