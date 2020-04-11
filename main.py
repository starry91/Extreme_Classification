import sys
from losses import Loss
from model.net import AttentionModel
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
import gzip
import logging
import time
from utils import *
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import torch.nn as nn
import matplotlib.pyplot as plt
torch.manual_seed(1)

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

torch.set_default_tensor_type(torch.FloatTensor)


class Solver():
    def __init__(self, model, loss, outdim_size, params, device=torch.device('cpu')):
        self.model = model
        self.model.to(device)
        self.epoch_num = params['epoch_num']
        self.batch_size = params['batch_size']
        self.loss = loss
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=params['learning_rate'], weight_decay=params['reg_par'])
        self.device = device

        self.reg_par = params['reg_par']
        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("XML.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, X_train, Y_train, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint_path=''):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        checkpoint = checkpoint_path+"/checkpoint.model"
        X_data_new = np.array([list(range(X_train.shape[1]))]*X_train.shape[0])
        X_TfIdftensor = torch.from_numpy(X_train[:, :, None])
        X_train = torch.from_numpy(X_data_new)
        Y_train = torch.from_numpy(Y_train)
        X_train = X_train.to(self.device)
        Y_train = Y_train.type('torch.FloatTensor').to(self.device)
        X_TfIdftensor = X_TfIdftensor.type('torch.FloatTensor').to(self.device)
        data_size = X_train.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))

            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_tfidf = X_TfIdftensor[batch_idx]
                batch_X_train = X_train[batch_idx, :]
                batch_Y_train = Y_train[batch_idx, :]
                x_hidden, y_hidden, y_predicted = self.model(
                    batch_X_train, batch_tfidf, batch_Y_train)
                loss = self.loss(x_hidden, y_hidden,
                                 y_predicted, batch_Y_train)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))

        fig = plt.figure()
        plt.plot(np.array(train_losses), 'r')
        plt.savefig('loss.png')
        plt.close(fig)
        checkpoint_ = torch.load(checkpoint)['model_state_dict']
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def predict(self, X_test):
        X_data_new = np.array([list(range(X_test.shape[1]))]*X_test.shape[0])
        tfidf = torch.from_numpy(X_test)
        tfidf = tfidf[:, :, None].type('torch.FloatTensor').to(self.device)
        bow = torch.from_numpy(X_data_new).to(self.device)
        with torch.no_grad():
            self.model.eval()
            data_size = X_test.shape[0]
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            outputs1 = []
            for batch_idx in batch_idxs:
                batch_x1 = bow[batch_idx, :]
                batch_tfidf = tfidf[batch_idx]
                o1 = self.model.predict(batch_x1, batch_tfidf)
                outputs1.append(o1)
        outputs = torch.cat(outputs1, dim=0).cpu().numpy(),
        return outputs


if __name__ == '__main__':
    ############
    # Parameters Section

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    print("Using", torch.cuda.device_count(), "GPUs")

    # the size of the new space learned by the model (number of the new features)
    input_size = 120
    output_size = 101
    embedding_size = 100
    attention_layer_size = 50
    encoder_layer_size = 120
    hidden_layer_size = 100

    # the parameters for training the network
    params = dict()
    params['learning_rate'] = 1e-3
    params['epoch_num'] = 3
    params['batch_size'] = 1024
    params['reg_par'] = 1e-5

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    r1 = 5e-7
    m = 0.8
    lamda = 10
    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # end of parameters section
    ############

    # Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
    # Datasets get stored under the datasets folder of user's Keras folder
    # normally under [Home Folder]/.keras/datasets/
    X_train, Y_train = load_data(
        "/home/praveen/Desktop/iiith-assignments/ExtremeClassification/mediamill/mediamill-train.arff")
    X_test, Y_test = load_data(
        '/home/praveen/Desktop/iiith-assignments/ExtremeClassification/mediamill/mediamill-test.arff')
    # Building, training, and producing the new features by DCCA
    model = AttentionModel(input_size=input_size, embedding_size=embedding_size,
                           attention_layer_size=attention_layer_size, encoder_layer_size=encoder_layer_size,
                           hidden_layer_size=hidden_layer_size, output_size=output_size)
    loss_func = Loss(outdim_size=output_size, use_all_singular_values=use_all_singular_values,
                     device=device, r1=r1, m=m, lamda=lamda).loss
    solver = Solver(model=model, loss=loss_func,
                    outdim_size=output_size, params=params, device=device)
    solver.fit(X_train, Y_train, checkpoint_path=".")
    y_pred = solver.predict(X_test)
    print(np.unique(y_pred, return_counts=True))
    print("P@1: ", p_k(y_pred, Y_test, 1))
    print("P@3: ", p_k(y_pred, Y_test, 3))
    print("P@5: ", p_k(y_pred, Y_test, 5))
    d = torch.load('checkpoint.model')
    solver.model.load_state_dict(d['model_state_dict'])
    solver.model.parameters()
