# Sentiment Analysis of Restaurant Reviews
## Project Overview
This project performs sentiment analysis on restaurant reviews, classifying them as Positive or Negative using Machine Learning and Natural Language Processing (NLP) techniques. The model is trained using TF-IDF vectorization and Multinomial Naïve Bayes, helping restaurants analyze customer feedback effectively.

## Features
* Preprocesses text data (removes stopwords, punctuation, and applies TF-IDF vectorization).
* Uses Naïve Bayes for sentiment classification.
* Evaluates performance with accuracy, confusion matrix, and classification report.
* Visualizes sentiment distribution using Matplotlib & Seaborn.
* Allows custom sentiment prediction for new reviews.

## Dataset
The dataset (Restaurant_Reviews 1.csv) contains two columns:

* Review  : The customer’s feedback text.
* Liked: Sentiment label (1 = Positive, 0 = Negative).
## Technologies Used
* Python
* Pandas, NumPy – Data manipulation
* NLTK, Sklearn – Text preprocessing & Machine Learning
* TfidfVectorizer – Text feature extraction
* Matplotlib, Seaborn – Data visualization
## Setup & Installation
Clone the repository or download the project files.
Install dependencies:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```
Run the script:
```bash
python sentiment_analysis.py
```
## Results & Insights
* Accuracy Score: Evaluates model performance.
* Confusion Matrix & Classification Report: Shows precision, recall, and F1-score.
* Graphical Representation: Displays distribution of positive and negative reviews.
### Example Sentiment Prediction
* Input: "The food was absolutely wonderful!"
* Output: Positive

 ## Future Enhancements
* Use Deep Learning (LSTMs, BERT) for improved accuracy.
* Extend support to multiple languages.
* Deploy as a web-based application for real-time feedback analysis.

This Sentiment Analysis Project helps businesses understand customer feedback and make data-driven improvements for better customer satisfaction! 

## Screenshots of output:

![Screenshot 2025-01-24 174233](https://github.com/user-attachments/assets/6aeb0afb-6db2-44a6-973f-40b67146545c)

![Screenshot 2025-01-24 174345](https://github.com/user-attachments/assets/d4c93f3f-aa43-4903-bb98-432eb75eb7c2)

![Screenshot 2025-01-24 174432](https://github.com/user-attachments/assets/a2eaf37c-ef89-4371-924a-63c3739bc3c7)


