import torch
from torch import nn
from torchsummary import summary
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from torch import optim
import numpy as np
from torchviz import make_dot
from scipy.io.arff import loadarff
import math
import os
import sys
import matplotlib.pyplot as plt
torch.manual_seed(1)
np.random.seed(1)


class FeatureEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, attention_layer_size, hidden_layer_size):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        # Attention Module
        self.attentionfc1 = nn.Linear(embedding_size,  attention_layer_size)
        self.attentionfc2 = nn.Linear(attention_layer_size, embedding_size)
        # fully connected layer
        self.featurefc1 = nn.Linear(embedding_size, hidden_layer_size)

    def forward(self, bow, tfidf):
        word_embedding = self.embedding(bow)
        word_embedding = tfidf * word_embedding
        attention_embedding = word_embedding
        attention_embedding = torch.relu(
            self.attentionfc1(attention_embedding))
        attention_embedding = torch.sigmoid(
            self.attentionfc2(attention_embedding))
        word_embedding = attention_embedding * word_embedding
        word_embedding = word_embedding.mean(1)
        x_hidden = torch.relu(self.featurefc1(word_embedding))
        return x_hidden


class Encoder(nn.Module):
    def __init__(self, output_size, encoder_layer_size, hidden_layer_size):
        super(Encoder, self).__init__()
        self.encoderfc1 = nn.Linear(output_size, encoder_layer_size)
        self.encoderfc2 = nn.Linear(encoder_layer_size, hidden_layer_size)

    def forward(self, labels):
        y_hidden = torch.relu(self.encoderfc1(labels))
        y_hidden = torch.relu(self.encoderfc2(y_hidden))
        return y_hidden


class Decoder(nn.Module):
    def __init__(self, output_size, encoder_layer_size, hidden_layer_size):
        super(Decoder, self).__init__()
        self.decoderfc1 = nn.Linear(hidden_layer_size, encoder_layer_size)
        self.decoderfc2 = nn.Linear(encoder_layer_size, output_size)

    def forward(self, y_hidden):
        y_predicted = torch.relu(self.decoderfc1(y_hidden))
        y_predicted = torch.sigmoid(self.decoderfc2(y_predicted))
        return y_predicted


class AttentionModel(nn.Module):
    def __init__(self, input_size, embedding_size, attention_layer_size, encoder_layer_size, hidden_layer_size, output_size):
        super(AttentionModel, self).__init__()
        self.featureEmbedding = FeatureEmbedding(
            input_size, embedding_size, attention_layer_size, hidden_layer_size)
        self.encoder = Encoder(
            output_size, encoder_layer_size, hidden_layer_size)
        self.decoder = Decoder(
            output_size, encoder_layer_size, hidden_layer_size)

    def forward(self, bow, tfidf, labels):
        x_hidden = self.featureEmbedding(bow, tfidf)
        y_hidden = self.encoder(labels)
        y_predicted = self.decoder(y_hidden)
        return x_hidden, y_hidden, y_predicted

    def predict(self, bow, tfidf):
        x_hidden = self.featureEmbedding(bow, tfidf)
        y_predicted = self.decoder(x_hidden)
        return y_predicted
