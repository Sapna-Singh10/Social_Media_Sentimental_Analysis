# Social Media Sentimental Analysis

This repository contains a project for performing multi-class sentiment analysis on social media data. The primary goal is to preprocess textual data and apply various machine learning models to classify the sentiment of a given text into a wide range of emotional categories.

## Dataset

The analysis is performed on the `cleaned_sentiment_data.csv` dataset. This dataset contains social media posts with associated metadata.

The key columns in the dataset include:
*   `Text`: The raw text content of the social media post.
*   `Sentiment`: The sentiment label for the text. This is a multi-class target with numerous categories like 'Positive', 'Negative', 'Anger', 'Joy', 'Sadness', etc.
*   `Timestamp`: The date and time the post was made.
*   `User`: The identifier for the user who made the post.
*   `Platform`: The social media platform (e.g., Twitter, Facebook, Instagram).
*   `Hashtags`: Hashtags associated with the post.
*   `Retweets`: The number of retweets.
*   `Likes`: The number of likes.
*   `Country`: The country of origin of the post.

## Methodology

The project follows a standard machine learning workflow for natural language processing tasks:

### 1. Exploratory Data Analysis (EDA)
*   The dataset is loaded and inspected for basic information, structure, and null values.
*   Visualizations are created to understand the data distribution, including:
    *   A count plot to show the distribution of different sentiment classes.
    *   A histogram to analyze the distribution of text lengths.

### 2. Text Preprocessing
A comprehensive text cleaning and preprocessing pipeline is applied to the raw text data to prepare it for modeling:
*   **Lowercasing**: All text is converted to lowercase.
*   **Noise Removal**: URLs, user mentions (@), hashtags (#), punctuation, and numerical digits are removed.
*   **Tokenization**: Text is split into individual words (tokens).
*   **Stopword Removal**: Common English stopwords (e.g., "a", "the", "is") are removed using the NLTK library.
*   **Lemmatization**: Words are reduced to their base or root form (e.g., "running" to "run") using NLTK's `WordNetLemmatizer`.

### 3. Feature Engineering
*   **TF-IDF Vectorization**: The preprocessed text is converted into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. This technique reflects how important a word is to a document in a collection or corpus. `TfidfVectorizer` from scikit-learn is used with a maximum of 3000 features.

### 4. Model Training and Comparison
Several machine learning models were trained on the TF-IDF features to perform the multi-class sentiment classification. The performance of each model was evaluated and compared. The models include:
*   Support Vector Machine (LinearSVC)
*   Random Forest Classifier
*   K-Nearest Neighbors (KNN)
*   Logistic Regression
*   Multinomial Naive Bayes

## Results

The models were evaluated based on accuracy, precision, recall, and F1-score. Due to the high number of distinct sentiment classes, the task is inherently complex.

### Model Performance Comparison

The table below summarizes the performance of the different models on the test set.

| Model                 | Accuracy | Precision (macro avg) | Recall (macro avg) | F1-score (macro avg) |
| --------------------- | :------: | :-------------------: | :----------------: | :------------------: |
| **SVM (LinearSVC)**   |  0.4286  |        0.2873         |       0.3033       |        0.2788        |
| **Random Forest**     |  0.4150  |        0.3506         |       0.3252       |        0.3192        |
| **K-Nearest Neighbors**|  0.3810  |        0.2644         |       0.2858       |        0.2608        |
| **Logistic Regression**|  0.3129  |        0.2383         |       0.2314       |        0.2220        |
| **Naive Bayes**       |  0.1429  |        0.0304         |       0.0409       |        0.0258        |

The Support Vector Machine (LinearSVC) and Random Forest models demonstrated the best performance, with LinearSVC achieving the highest accuracy of approximately 42.9%.

### Visualizations

The analysis includes several visualizations to interpret the data and results:

*   **Positive Sentiment Word Cloud**: A word cloud was generated to visualize the most frequent words in posts with a 'positive' sentiment.
*   **Model Comparison Plots**: Bar charts were created to visually compare the Accuracy and F1-Scores of the trained models, providing a clear view of their relative performance.

## Getting Started

To run this project locally, follow these steps:

### Prerequisites

Ensure you have Python 3 installed. You will also need the following libraries:
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   nltk
*   wordcloud
*   textblob
*   scikit-learn

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/sapna-singh10/Social_Media_Sentimental_Analysis.git
    cd Social_Media_Sentimental_Analysis
    ```

2.  Install the required Python packages:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk wordcloud textblob scikit-learn
    ```

3.  Download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

### Usage

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open and run the `Untitled.ipynb` notebook to see the full analysis, from data loading to model comparison.
