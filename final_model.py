import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Import libraries

# Step 2: Load the data
data = pd.read_excel('data.xlsx')  # Replace 'data.xlsx' with the path to your Excel file
sentences = data['Sentence']
labels = data['Label']

# Step 3: Preprocess the data (optional, customize as per your requirements)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Step 5: Extract features from the text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 6: Train a machine learning model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Step 7: Fine-tune the model (optional, customize as per your requirements)

# Step 8: Make predictions
sentence = "This is a test sentence."
sentence_vectorized = vectorizer.transform([sentence])
predicted_label = model.predict(sentence_vectorized)[0]
print("Predicted label:", predicted_label)

# Step 9: Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 10: Deploy and use the model (customize as per your requirements)