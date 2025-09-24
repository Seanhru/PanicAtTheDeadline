"""
Group: Panic at the Deadline
Members: Jessica Bailey, Ariana Elahi, Sean Bamfo
Date: 9/24/2025
Project: Character Line Classification from South Park Scripts
---------------------------------------------------------------
This script builds a machine learning model to classify which South Park
character spoke a given line. The workflow includes:

1. Data loading and preprocessing
2. Text cleaning and balancing (undersampling and oversampling)
3. Feature extraction using TF-IDF (with optional numeric features)
4. Model training with Logistic Regression
5. Model evaluation using precision, recall, F1, and confusion matrices
6. Visualization of classification performance
7. Identification of important words/phrases for each character

Libraries:
- pandas and numpy for data handling
- scikit-learn for feature extraction, preprocessing, model training, evaluation
- matplotlib and seaborn for visualization
- scipy for matrix operations
"""

# ----------------------
# Import Required Libraries
# ----------------------
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections

#%% 1. Load and filter top 10 characters
# Read dataset from CSV
df = pd.read_csv("All-seasons.csv")

# Identify the 10 characters with the most lines
top_characters = df['Character'].value_counts().nlargest(10).index

# Keep only rows where the character is one of the top 10
df = df[df['Character'].isin(top_characters)]

#%% 2. Document dataset
# Display dataset statistics
print("Total rows:", df.shape[0])  # number of rows
print("Distinct episodes:", df[['Season','Episode']].drop_duplicates().shape[0])  # unique episodes
print("Seasons covered:", df['Season'].unique())  # list of unique seasons

#%% 3. Clean text
# Convert lines to lowercase and strip whitespace
df['Line'] = df['Line'].str.lower().str.strip()

# Remove punctuation from lines
df['Line'] = df['Line'].str.translate(str.maketrans('', '', string.punctuation))

# Drop rows with missing values in 'Line' or 'Character'
df = df.dropna(subset=['Line','Character'])

#%% 3.5 Balance dataset
# To prevent imbalance, perform undersampling and oversampling

# ---- Undersampling ----
df_balanced = pd.DataFrame()
min_count = df['Character'].value_counts().min()  # size of smallest class

# Randomly sample each character down to the min_count
for character in df['Character'].unique():
    subset = df[df['Character'] == character]
    subset_downsampled = resample(
        subset, replace=False, n_samples=min_count, random_state=42
    )
    df_balanced = pd.concat([df_balanced, subset_downsampled])

# Shuffle dataset
df = df_balanced.sample(frac=1, random_state=42)
print("Balanced counts (undersampling):\n", df['Character'].value_counts())

# ---- Oversampling ----
df_balanced = pd.DataFrame()
max_count = df['Character'].value_counts().max()  # size of largest class

# Randomly sample each character up to the max_count
for character in df['Character'].unique():
    subset = df[df['Character'] == character]
    subset_upsampled = resample(
        subset, replace=True, n_samples=max_count, random_state=42
    )
    df_balanced = pd.concat([df_balanced, subset_upsampled])

# Shuffle dataset
df = df_balanced.sample(frac=1, random_state=42)
print("Balanced counts (oversampling):\n", df['Character'].value_counts())

# ---- Save cleaned, filtered, and balanced dataset ----
# This creates a new CSV file that contains only:
# - Top 10 characters
# - Cleaned dialogue lines
# - Balanced representation across characters
df.to_csv("cleaned_dataset.csv", index=False)
print("Saved cleaned dataset to 'cleaned_dataset.csv'")

#%% 4. TF-IDF vectorization (1-3 grams)
# Transform text into numerical features (word importance scores)
# Using unigrams, bigrams, trigrams (1-3 grams), with constraints
vectorizer = TfidfVectorizer(
    ngram_range=(1,3), max_features=30000, min_df=2, max_df=0.95
)
X_text = vectorizer.fit_transform(df['Line'])  # sparse matrix of TF-IDF features
y = df['Character']  # target labels

#%% 5. Add numeric features (Season, Episode)
# Normalize season/episode numbers and add as features
numeric_features = df[['Season','Episode']].copy()
scaler = StandardScaler()
X_numeric = scaler.fit_transform(numeric_features)

# Combine text features (sparse matrix) with numeric features
X = hstack([X_text, X_numeric])

#%% 6. Train/Validation/Test Split
# Split data into train, validation, and test sets with stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

#%% 7. Train Logistic Regression
# Train multinomial logistic regression classifier
model = LogisticRegression(
    max_iter=2000,
    multi_class='multinomial',
    n_jobs=-1,   # parallel processing
    C=10.0       # regularization strength (inverse)
)
model.fit(X_train, y_train)

#%% 8. Evaluate
# Predictions on validation set
y_pred = model.predict(X_val)

# Print precision, recall, F1 scores
print(classification_report(y_val, y_pred))

# ---- Confusion matrix ----
cm = confusion_matrix(y_val, y_pred, labels=top_characters)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=top_characters,
            yticklabels=top_characters,
            cmap='Blues')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ---- Precision, Recall, F1 Heatmap ----
metrics = precision_recall_fscore_support(
    y_val, y_pred, labels=top_characters, zero_division=0
)
report_df = pd.DataFrame(
    metrics, index=["Precision","Recall","F1","Support"], columns=top_characters
).T

plt.figure(figsize=(10,6))
sns.heatmap(report_df.iloc[:,:3], annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Precision, Recall, and F1 by Character")
plt.show()

# ---- Per-class accuracy ----
acc_per_class = (y_val == y_pred).groupby(y_val).mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=acc_per_class.index, y=acc_per_class.values, palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.show()

# ---- Most common misclassifications ----
misclassified = [(true,pred) for true,pred in zip(y_val, y_pred) if true != pred]
mis_counts = collections.Counter(misclassified).most_common(10)
mis_df = pd.DataFrame(mis_counts, columns=["(True, Predicted)","Count"])

plt.figure(figsize=(10,5))
sns.barplot(x="Count", y="(True, Predicted)", data=mis_df, palette="magma")
plt.title("Most Common Misclassifications")
plt.show()

#%% 9. Feature importance (top words/phrases per character)
# Identify the most influential words/phrases for each character
feature_names = vectorizer.get_feature_names_out()
for i, class_label in enumerate(model.classes_):
    top_indices = np.argsort(model.coef_[i])[-10:]  # top 10 features
    print(f"Top words/phrases for {class_label}: {[feature_names[j] for j in top_indices]}")
