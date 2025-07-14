'''
project name    : SMS Spam Detector
designed by     : Dinesh Kumar G
designed on     : 13.07.2025 , 10:27pm
purpose         :Creating a ML model to detect spam SMS
'''
# Import libraries
import pandas as pd
import string
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df

#Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

#Apply cleaning
df['cleaned_message'] = df['message'].apply(clean_text)

#TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['cleaned_message'])

#Target labels
y = df['label']

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

#Predict on test data
y_pred = model.predict(X_test)
y_pred

#Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Classification Report Bar Chart
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot precision, recall, F1-score for ham and spam
report_df[['precision', 'recall', 'f1-score']].iloc[:2].plot(kind='bar', figsize=(7, 4))
plt.title("Precision, Recall & F1-score for Ham & Spam")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.show()

#Compare Actual vs Predicted
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Message': df.loc[y_test.index, 'message'].values
})
print("\ Actual vs Predicted:")
print(comparison_df.head(10))

# Show incorrect predictions
print("\n Incorrect Predictions:")
print(comparison_df[comparison_df['Actual'] != comparison_df['Predicted']].head(5))

#Try a sample SMS
sample_message = ["You won a FREE entry to a prize draw! Click here."]
sample_cleaned = [clean_text(sample_message[0])]
sample_vector = tfidf.transform(sample_cleaned)
prediction = model.predict(sample_vector)

print("\n Sample Message Prediction:", "Spam" if prediction[0] == 1 else "Ham")
