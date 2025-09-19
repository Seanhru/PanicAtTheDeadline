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

#%% 4. TF-IDF vectorization (1-3 grams)
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=10000, min_df=3, max_df=0.9)
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
    class_weight='balanced',
    n_jobs=-1
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

#%% 9. Feature importance (top words/phrases per character)
import numpy as np
feature_names = vectorizer.get_feature_names_out()
for i, class_label in enumerate(model.classes_):
    top_indices = np.argsort(model.coef_[i])[-10:]
    print(f"Top words/phrases for {class_label}: {[feature_names[j] for j in top_indices]}")
