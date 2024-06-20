# Stock Embeddings Generation Repository
## Overview
This repository focuses on applying machine learning models to generate and analyze stock embeddings. Our goal is to provide tools for visualizing financial data and capturing market dynamics through advanced NLP techniques and context-aware algorithms.

## Contents

### Python Notebooks
- **pointwise_embeddings.ipynb**
  - Demonstrates the generation of pointwise stock embeddings and includes training and evaluation methods.
- **window_embeddings.ipynb**
  - Focuses on creating embeddings using a sliding window over the stock data, capturing temporal context.
- **CS521.ipynb**
  - Contains advanced financial machine learning techniques, as part of the CS521 course content.

### Python Scripts
- **training_helpers.py**
  - Provides batch processing, training loop management, and evaluation metrics for embedding models.
- **visualisation_functions.py**
  - Tools for visualizing stock market data and interpreting embedding model performance.
- **window_context.py**
  - Manages data within a window context for models requiring time-based historical input.
- **returns_data_class.py**
  - Processes and normalizes financial returns data for model input.
- **sector_classification.py**
  - Implements stock classification into market sectors based on embedding characteristics.
- **embedding_models.py**
  - Contains the implementation of various stock embedding models and associated utilities.
- **base_model.py**
  - Outlines a standard model structure used across the project's machine learning models.
- **pointwise_context.py**
  - Enhances stock data input with additional context for improved model accuracy.
## Installed Packages and Their Uses

In this project, we utilize a range of Python packages, each serving a specific role in the process of generating, analyzing, and visualizing stock embeddings:
- **yfinance**: A popular library that offers a reliable, threaded, and Pythonic way to download historical market data from Yahoo Finance. It is often used for fetching historical stock price data, as well as for obtaining information on dividends, stock splits, and other aspects of securities.
- **numpy**: A fundamental package for scientific computing with Python. It provides support for arrays, matrices, and high-level mathematical functions to operate on these data structures.
- **torch**: Also known as PyTorch, it is an open-source machine learning library used for applications such as computer vision and natural language processing, primarily focused on deep learning.
- **tqdm**: A simple yet powerful tool that provides a fast, extensible progress bar for loops and code blocks, enhancing the visibility of the processing stages.
- **sklearn**: Short for scikit-learn, this library offers simple and efficient tools for predictive data analysis. It is accessible to everybody and reusable in various contexts, built on NumPy, SciPy, and matplotlib.
- **SMOTE**: (Synthetic Minority Over-sampling Technique) and `imblearn.over_sampling`: Part of the imbalanced-learn package, these are used for over-sampling minority classes in the dataset by generating synthetic examples to create a balanced class distribution.
- **pandas**: An essential data analysis and manipulation tool, providing data structures and operations for manipulating numerical tables and time series.
- **plotly.express**: A terse, consistent, high-level API for creating figures. It is used for building complex visualizations with simple one-liner code.
- **PCA** (from sklearn.decomposition): PCA, or Principal Component Analysis, is a dimensionality-reduction method used to reduce the complexity of data while preserving as much information as possible.
- **TSNE** (from sklearn.manifold): t-Distributed Stochastic Neighbor Embedding (t-SNE) is a tool to visualize high-dimensional data by reducing it to two or three dimensions while keeping similar instances close to each other.
- **ABC, abstractmethod** (from abc): These are used to create abstract base classes and define abstract methods within them, which is a way of creating a blueprint for other classes.
- **scipy.spatial.distance**: Provides functions to compute distances between points or datasets. Specifically, `pdist` computes pairwise distances between observations, and `squareform` is used to format the distance matrix.

The combination of these libraries forms the backbone of our data processing, model training, and visualization pipeline, supporting the sophisticated analysis required in financial market applications.


## Main NLP Model
### Architecture
- The primary NLP model uses a **Transformer-based architecture** with attention mechanisms to contextualize data points in stock market sequences.

### Key Functionalities
- **Embedding Generation**
  - Generates semantic-rich embeddings from stock data.
- **Contextual Analysis**
  - Analyzes stock data within given market contexts for trend analysis and anomaly detection.
- **Sector Classification**
  - Classifies stocks into industry sectors to predict market trends.

### Data Processing

1. **Preprocessing**
   - Cleansing and normalizing financial data.
2. **Vectorization**
   - Converting data into numerical form suitable for model input.
3. **Attention Mechanisms**
   - Applying self-attention to prioritize relevant data points dynamically.

## Model Outputs and Evaluation

### Pointwise Embedding Model Output

- **Performance Metrics**
  - Precision Score: **0.68**
  - Recall Score: **0.65**
  - F1 Score: **0.65**
  - Accuracy Score: **0.65**
  - Accuracy Score Top-3: **0.85**

- **Visualization**
  - PCA scatter plot demonstrating sector clustering of stock embeddings.
    ![Pointwise Embedding Scatter Plot](utils/Pointwise_embedding.jpg "Pointwise Embedding Scatter Plot")

### Window-wise Embedding Model Output

- **Performance Metrics**
  - Precision Score: **0.52**
  - Recall Score: **0.48**
  - F1 Score: **0.49**
  - Accuracy Score: **0.48**

- **Visualization**
  - PCA scatter plot showing dispersion and sector overlap in stock embeddings.
    ![Window-wise Embedding Scatter Plot](utils/Windowwise_embedding.jpg "Window-wise Embedding Scatter Plot")

