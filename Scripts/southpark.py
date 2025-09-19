import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Load your CSV
df = pd.read_csv("All-seasons.csv")

# Optionally filter to top 10 characters first (if you want only main cast)
top10 = df["Character"].value_counts().head(10).index
df = df[df["Character"].isin(top10)]

# Features and labels
X = df["Line"]
y = df["Character"]

# Split train/test (stratify to preserve character distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF with n-grams
vectorizer = TfidfVectorizer(
    max_features=20000,     # more features = better capture of vocabulary
    ngram_range=(1, 3),     # unigrams + bigrams + trigrams
    min_df=2,               # ignore super-rare phrases
    sublinear_tf=True       # log-scaling of term frequency
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Classifier (SVM often stronger than Logistic Regression for text)
clf = LinearSVC(class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
