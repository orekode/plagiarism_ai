import joblib  # For loading the saved model
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF vectorization
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string


tfidf_vectorizer = TfidfVectorizer()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Load the saved model
loaded_model = joblib.load('plagiarism_model.pkl')

# New text for plagiarism detection
new_text = "A new group of researchers found out a new class of butterfly in the Amazon forest of rain."

# Preprocess the new text (e.g., apply the same preprocessing steps as during training)
new_text = preprocess_text(new_text)

# Convert the preprocessed text into TF-IDF vectors (assuming you have the vectorizer)
new_text_vector = tfidf_vectorizer.transform([new_text])

# Make predictions using the loaded model
prediction = loaded_model.predict(new_text_vector)

# Calculate cosine similarity between new text and training data
cosine_similarity_score = cosine_similarity(new_text_vector, X_train).max()

# Interpret the prediction and similarity score
if prediction[0] == 0:
    print("The text is not plagiarized.")
else:
    print(f"The text is plagiarized with a similarity score of {cosine_similarity_score*100:.2f}%.")