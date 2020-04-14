
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


class Loss():
    def __init__(self, outdim_size, use_all_singular_values, device, r1, m, lamda):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        self.r1 = r1
        self.m = m
        self.lamda = lamda

    def reconstructingLoss(self, y_predicted, y_actual):
        loss = 0
        for i in range(y_predicted.shape[0]):
            y_hat_n = y_predicted[i][y_actual[i] == 0]
            y_hat_p = y_predicted[i][y_actual[i] == 1]
            if(len(y_hat_p) == 0):
                continue
            y_p_min = torch.min(y_hat_p)
            y_n_max = torch.max(y_hat_n)
            loss1 = torch.sum(y_hat_n+self.m-y_p_min)
            loss1[loss1 < 0] = 0
            loss2 = torch.sum(-y_hat_p+self.m+y_n_max)
            loss2[loss2 < 0] = 0
            loss += (loss1+loss2)
        loss /= y_predicted.shape[0]
        # print("ReconstrLoss",loss)
        return loss

    def hidden_dcca(self, H1, H2):
        r1 = self.r1
        r2 = 1e-3
        eps = 1e-12

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat12).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        # posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        # D1 = D1[posInd1]
        # V1 = V1[:, posInd1]
        # posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        # D2 = D2[posInd2]
        # V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
                # all singular values are used to calculate the correlation
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            # print(tmp)
            corr = torch.sqrt(tmp)
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            U, V = torch.symeig(torch.matmul(
                Tval.t(), Tval), eigenvectors=True)
            print("U", U)
            U = torch.abs(U)
            # U = U[torch.gt(U, eps).nonzero()[:, 0]]
            U = U.topk(self.outdim_size)[0]
            # print("Utop",U)
            # print("Usqrt",torch.sqrt(U))
            # print("Usum",torch.sum(torch.sqrt(U)))
            corr = torch.sum(torch.sqrt(U))
        # print("corr",-corr)
        return 100-corr

    def hiddenLoss(self, x_hidden, y_hidden):
        temp = torch.sum((x_hidden-y_hidden)**2)
        temp /= x_hidden.shape[0]
        return temp

    def loss(self, x_hidden, y_hidden, y_predicted, y_actual, print_flag=False):
        loss_hidden = self.hiddenLoss(x_hidden, y_hidden)
        loss_ae = self.reconstructingLoss(y_predicted, y_actual)
        # print("loss_hidden: ", loss_hidden)
        # print("loss_ae: ", loss_ae)
        if print_flag:
            print("Hidden loss = {0}, Reconstruction Loss = {1}".format(
                loss_hidden, loss_ae))
        return (loss_hidden+self.lamda*loss_ae)

    def dccaLoss(self, x_hidden, y_hidden, y_predicted, y_actual, print_flag=False):
        loss_hidden = self.hidden_dcca(x_hidden, y_hidden)
        loss_ae = self.reconstructingLoss(y_predicted, y_actual)
        if print_flag:
            print("Hidden loss = {0}, Reconstruction Loss = {1}".format(
                loss_hidden, loss_ae))
        return (loss_hidden+self.lamda*loss_ae)
