import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance as distance
import numpy as np
import os
import pickle

class Plagiarism:
    def __init__(self, model_path='similarity_model.pt', embeddings_path='training_embeddings.pkl'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # self.model.eval()
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

    def train(self, descriptions):
        new_embeddings = [self.encode_description(desc) for desc in descriptions]
        self.training_embeddings.extend(new_embeddings)
        self.save_embeddings()
        self.save_model()

    def calculate_similarity(self, new_description):
        new_embedding = self.encode_description(new_description)
        # print([new_embedding], self.training_embeddings)
        similarities = cosine_similarity([new_embedding], self.training_embeddings)
        return max(10, (similarities.min() * 100) )  # Return the highest similarity percentage
    

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))

    def save_embeddings(self):
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.training_embeddings, f)

    def load_embeddings(self):
        with open(self.embeddings_path, 'rb') as f:
            self.training_embeddings = pickle.load(f)


checker = Plagiarism()
# checker.train(['A building that can walk and the moon and breathe fire.'])
# print(checker.calculate_similarity('This project aims to create a cutting-edge e-commerce platform dedicated to promoting sustainable fashion brands. The platform will serve as a marketplace where eco-conscious consumers can discover and purchase clothing, accessories, and footwear made from sustainable materials and ethical manufacturing processes. The project will feature a rich product catalog, advanced search and filtering options, and personalized recommendations powered by AI. It will also include detailed sustainability metrics for each product, helping consumers make informed choices. Additionally, the platform will integrate social features, such as user reviews and community forums, to foster a community of like-minded individuals. The development will focus on responsive design, secure payment gateways, and scalable architecture to support a growing number of users and products.'))
# print(checker.calculate_similarity('A building that can walk and the moon and breathe fire.'))