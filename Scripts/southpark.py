import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns


#%% 1. Load and filter top 10 characters
df = pd.read_csv("All-seasons.csv")
top_characters = df['Character'].value_counts().nlargest(10).index
df = df[df['Character'].isin(top_characters)]

#%% 2. Document dataset
print("Total rows:", df.shape[0])
print("Distinct episodes:", df[['Season','Episode']].drop_duplicates().shape[0])
print("Seasons covered:", df['Season'].unique())

#%% 3. Clean text
df['Line'] = df['Line'].str.lower().str.strip()
df['Line'] = df['Line'].str.translate(str.maketrans('', '', string.punctuation))
df = df.dropna(subset=['Line','Character'])

#%% 3.5 Balance dataset (undersampling)
from sklearn.utils import resample

df_balanced = pd.DataFrame()
min_count = df['Character'].value_counts().min()


for character in df['Character'].unique():
   subset = df[df['Character'] == character]
   subset_downsampled = resample(
       subset,
       replace=False,
       n_samples=min_count,
       random_state=42
   )
   df_balanced = pd.concat([df_balanced, subset_downsampled])


df = df_balanced.sample(frac=1, random_state=42)  # shuffle
print("Balanced counts:\n", df['Character'].value_counts())

#Oversample
df_balanced = pd.DataFrame()
max_count = df['Character'].value_counts().max()


for character in df['Character'].unique():
   subset = df[df['Character'] == character]
   subset_upsampled = resample(
       subset,
       replace=True,
       n_samples=max_count,
       random_state=42
   )
   df_balanced = pd.concat([df_balanced, subset_upsampled])


df = df_balanced.sample(frac=1, random_state=42)  # shuffle
print("Balanced counts:\n", df['Character'].value_counts())

#%% 4. TF-IDF vectorization (1-3 grams)
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=30000, min_df=2, max_df=0.95)
X_text = vectorizer.fit_transform(df['Line'])
y = df['Character']

#%% 5. Optional: add numeric features (Season, Episode)
numeric_features = df[['Season','Episode']].copy()
scaler = StandardScaler()
X_numeric = scaler.fit_transform(numeric_features)

# Combine TF-IDF and numeric features
from scipy.sparse import hstack
X = hstack([X_text, X_numeric])

#%% 6. Train/Validation/Test Split (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

#%% 7. Train Logistic Regression with class balancing
model = LogisticRegression(
    max_iter=2000,
    multi_class='multinomial',
    #class_weight='balanced',
    n_jobs=-1,
    C=10.0
)
model.fit(X_train, y_train)

#%% 8. Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=top_characters)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=top_characters, yticklabels=top_characters, cmap='Blues')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_recall_fscore_support

metrics = precision_recall_fscore_support(y_val, y_pred, labels=top_characters, zero_division=0)
report_df = pd.DataFrame(metrics, index=["Precision","Recall","F1","Support"], columns=top_characters).T

plt.figure(figsize=(10,6))
sns.heatmap(report_df.iloc[:,:3], annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Precision, Recall, and F1 by Character")
plt.show()

acc_per_class = (y_val == y_pred).groupby(y_val).mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=acc_per_class.index, y=acc_per_class.values, palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.show()

import collections
misclassified = [(true,pred) for true,pred in zip(y_val, y_pred) if true != pred]
mis_counts = collections.Counter(misclassified).most_common(10)
mis_df = pd.DataFrame(mis_counts, columns=["(True, Predicted)","Count"])

plt.figure(figsize=(10,5))
sns.barplot(x="Count", y="(True, Predicted)", data=mis_df, palette="magma")
plt.title("Most Common Misclassifications")
plt.show()


#%% 9. Feature importance (top words/phrases per character)
import numpy as np
feature_names = vectorizer.get_feature_names_out()
for i, class_label in enumerate(model.classes_):
    top_indices = np.argsort(model.coef_[i])[-10:]
    print(f"Top words/phrases for {class_label}: {[feature_names[j] for j in top_indices]}")
