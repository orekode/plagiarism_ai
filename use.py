import torch
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class UseModel:
    def __init__(self, model_path='similarity_model.pt', embeddings_path='training_embeddings.pkl'):
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.training_embeddings = []

        # Load the model and embeddings if they exist
        if os.path.exists(model_path):
            self.load_model()
        if os.path.exists(embeddings_path):
            self.load_embeddings()

    def encode_description(self, description):
        inputs = self.tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def calculate_similarity(self, new_description):
        new_embedding = self.encode_description(new_description)
        print([new_embedding], self.training_embeddings)
        similarities = cosine_similarity([new_embedding], self.training_embeddings)
        return max(10, (similarities.min() * 100) - 40)

    def load_model(self):
        self.model = torch.load(self.model_path)

    def load_embeddings(self):
        with open(self.embeddings_path, 'rb') as f:
            self.training_embeddings = pickle.load(f)


checker = UseModel()
print(checker.calculate_similarity('A building that can walk and the moon and breathe fire.'))