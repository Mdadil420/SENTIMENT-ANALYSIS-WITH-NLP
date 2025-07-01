# Task-2: Sentiment Analysis using TF-IDF and Logistic Regression



# Step 2: Sample Dataset (You can replace this with your own CSV)
data = {
    'review': [
        'I love this product, it is amazing!',
        'Worst experience ever.',
        'Not bad, could be better.',
        'Absolutely fantastic service!',
        'I hate it.',
        'It was okay, nothing special.',
        'Excellent quality and great value.',
        'Terrible, will never buy again.'
    ],
    'sentiment': [1, 0, 0, 1, 0, 0, 1, 0]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# Step 3: TF-IDF Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
