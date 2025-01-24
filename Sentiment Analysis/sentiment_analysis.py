import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('Restaurant_Reviews 1.csv')

# Map numerical labels to text labels
data['Liked'] = data['Liked'].map({1: 'Positive', 0: 'Negative'})

# Display dataset info
print(f"Dataset loaded with {data.shape[0]} entries.")
print(data.head())

# Plot the distribution of Positive & Negative reviews
plt.figure(figsize=(6, 4))
sns.countplot(x=data['Liked'], palette=['#ff6347', '#4682b4'])
plt.title("Distribution of Positive & Negative Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Split the data into training and testing sets
X = data['Review']
y = data['Liked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Function for predicting sentiment of new reviews
def predict_sentiment(review):
    review_tfidf = tfidf_vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    return prediction

# Example usage
new_review = "The food was absolutely wonderful!"
sentiment = predict_sentiment(new_review)
print(f"The sentiment for the review '{new_review}' is: {sentiment}")
