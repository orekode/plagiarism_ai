import nltk
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
from nltk.corpus import stopwords
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import random

class PlagiarismDetector:
    def __init__(self, model_path='plagiarism_model.pkl'):
        self.model_path = model_path
        self.model = self.load_model()
        self.vectorizer = TfidfVectorizer()
        self.projects = [
            "Develop an AI-powered, underwater, virtual reality experience that simulates the migration patterns of fictional, gelatinous creatures called Gloopernaughts as they travel through a wormhole in search of a new planetary habitat.",
            "Design a wearable, bioluminescent device that uses real-time environmental data to generate a personalized, algorithmic soundtrack that harmonizes with the user's brain waves, while also cultivating a miniature, self-sustaining ecosystem on the user's wrist.",
            "Create a swarm intelligence-based, autonomous farming system that utilizes fractal geometry to optimize crop yields, while also deploying a network of micro-drones to pollinate plants using a proprietary, AI-generated, species-specific bee dance language.",
            "Develop a neural network-powered, generative art platform that uses real-time astronomical data to create immersive, psychedelic visualizations of distant nebulae, while also generating a corresponding, AI-composed soundtrack that adapts to the user's emotional response.",
            "Design a time-traveling, culinary experience platform that uses machine learning to predict and recreate historical, cultural dishes from ancient civilizations, while also generating a virtual, 3D environment that simulates the sights, sounds, and aromas of the original culinary context."
        ]

    def load_model(self):
        try:
            return joblib.load(self.model_path)
        except FileNotFoundError:
            return LogisticRegression()

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def preprocess_text(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        text = " ".join(word for word in text.split() if word not in stop_words)
        return text

    def create_plagiarized_versions(self, text, num_versions=5):
        plagiarized_versions = []
        words = text.split()
        for _ in range(num_versions):
            plagiarized_version = []
            for word in words:
                if random.random() < 0.2:  # 20% chance of replacing a word
                    synonyms = self.get_synonyms(word)
                    if synonyms:
                        plagiarized_version.append(random.choice(synonyms))
                    else:
                        plagiarized_version.append(word)
                else:
                    plagiarized_version.append(word)
            plagiarized_versions.append(" ".join(plagiarized_version))
        return plagiarized_versions

    def get_synonyms(self, word):
        synonyms = set()
        for syn in nltk.corpus.wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def train(self, text, non_plagiarized_versions=[]):

        plagiarized_versions = self.create_plagiarized_versions(text)

        if len(non_plagiarized_versions) < 5:
            non_plagiarized_versions = non_plagiarized_versions + self.projects[0:5 - len(non_plagiarized_versions)]
        

        data = pd.DataFrame({

            "text": plagiarized_versions + non_plagiarized_versions,

            "label": [1] * len(plagiarized_versions) + [0] * len(non_plagiarized_versions)

        })

        X = self.vectorizer.fit_transform(data["text"])

        y = data["label"]

        self.model.fit(X, y)

        self.save_model()

    def predict(self, text):
        data = pd.DataFrame({"source_text": [text]})
        data["source_text"] = data["source_text"].apply(self.preprocess_text)
        X = self.vectorizer.transform(data["source_text"])
        return self.model.predict(X)

    def evaluate(self, text):
        plagiarized_versions = self.create_plagiarized_versions(text)
        data = pd.DataFrame({"source_text": [text] * len(plagiarized_versions), "plagiarized_text": plagiarized_versions})
        data["source_text"] = data["source_text"].apply(self.preprocess_text)
        data["plagiarized_text"] = data["plagiarized_text"].apply(self.preprocess_text)
        X = self.vectorizer.transform(data["source_text"] + " " + data["plagiarized_text"])
        y = [1] * len(plagiarized_versions)  # all plagiarized versions are labeled as 1
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        classification_rep = classification_report(y, y_pred)
        return accuracy, classification_rep

# Example usage:
detector = PlagiarismDetector()

# Train the model
text = "This is a sample text."
detector.train(text)

# Use the model to predict
new_text = "something wayy different."
prediction = detector.predict(new_text)
print("similarity: ", prediction * 100)

