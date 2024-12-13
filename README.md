# NLP Project 3: Real vs. Fake News Classification

This project uses Natural Language Processing (NLP) techniques to classify news article headlines as either real or fake news. The goal is to build a robust machine learning model that accurately predicts the authenticity of news headlines based on textual features.


## Features

- **Text Preprocessing:** Includes tokenization, stopword removal, and lemmatization.
- **Feature Extraction:** Converts textual data into numerical representations using TF-IDF Vectorization.
- **Model Comparison:** Explored and evaluated several models including Decision Tree, Random Forest, Naive Bayes, and XGBoost.
- **Final Model:** Selected Logistic Regression for its superior performance, achieving a 93% accuracy on the training dataset.


## Technologies Used

This project utilizes the following libraries and tools:

- **Python:** For the end-to-end implementation.
- **Pandas and NumPy:** For data manipulation and numerical computations.
- **scikit-learn:** For feature extraction, model training, and evaluation.
- **NLTK:** For advanced text preprocessing, including tokenization and lemmatization.


## Dataset

The project uses the following CSV files:

- **training_data_lowercase.csv:** Contains labeled training data used to train the models.
- **testing_data_lowercase_nolabels.csv:** Contains unlabeled test data used for evaluation.
- **testing_data_lowercase_nolabels_G3.csv:** Contains test data with predicted labels from the trained model.


## Workflow Overview

1.	**Preprocessing:**
- Converted text to lowercase.
- Removed special characters and stopwords.
- Tokenized text and performed lemmatization using NLTK’s WordNetLemmatizer.

2.	**Feature Engineering:**
- Applied TF-IDF Vectorization to transform text into numerical data.

3.	**Model Training and Evaluation:**
- Experimented with multiple models:
- Decision Tree
- Random Forest
- Naive Bayes
- XGBoost
- Final Model: Logistic Regression with 93% accuracy on the training dataset.

4.	**Prediction:**
- Used the trained model to predict labels for test data in testing_data_lowercase_nolabels.csv.
- Results saved in testing_data_lowercase_nolabels_G3.csv.


## Results

The project achieved the following:
- **93% Accuracy** on the training dataset.
- Effective feature extraction using TF-IDF and preprocessing with NLP techniques.
- Logistic Regression selected as the final model for its balance of simplicity and performance.


## Project Structure

├── Archive/                  # Folder containing intermediate working files.  
├── NLPG3.pptx               # Team presentation  
├── main.ipynb                # Final implementation notebook.  
├── training_data_lowercase.csv  
├── testing_data_lowercase_nolabels.csv  
├── testing_data_lowercase_nolabels_G3.csv  
└── README.md                 # Project documentation (this file).  