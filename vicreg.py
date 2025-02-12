import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.api import Sequential
from keras.api.layers import ReLU
from squark.layers import QBatchNormalization, QDense


class VICReg(nn.Module):
    def __init__(
        self,

        #a backbone extracts high-level features from input data and gives a latent feature representation
        backbone: 'ModelBackbone',

        #a projector maps extracted features into a higher-dim space for VICReg loss
        projector,
        
        num_features,  # Size of the projection layer
        batch_size,
        sim_coeff=50,
        std_coeff=50,
        cov_coeff=1,
    ):
        super().__init__()
        self.num_features = num_features
        self.backbone = backbone
        self.projector = projector

        self.batch_size = batch_size

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        
    #Passes x and y through the backbone and projector, transforming them into feature embeddings. Expects two different 
    #views (x and y)
    def forward(self, x, y):

        #expands representations Y and Y' into Z and Z'
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        # invariance loss
        repr_loss = F.mse_loss(x, y)

        #covariance loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        # variance loss - enforces a minimum variance of 1, preventing feature collapse
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        #also covariance loss idk why out of order lmao -measures feature dependencies
        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(
            self.num_features
        )
        #removes diagonal elements and penalizes high off-diagonal values, ensuring decorrelation
        hgq_loss = sum(self.backbone.model.losses)

        #calculate the total loss
        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss + hgq_loss

        return loss, repr_loss, std_loss, cov_loss

#Excludes bias and normalization layers from regularization.
def exclude_bias_and_norm(p):
    return p.ndim == 1

#Extracts off-diagonal elements of a square matrix.
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

#Defines a quantization-aware neural network.
class ModelBackbone(nn.Module):
    #number of neaurons per layer = 8
    def __init__(self, nodes=[8]):
        super().__init__()

        self.blocks = nn.ModuleList()

        layers = []
        for n in nodes:
            #quantized dense layer ** look it up
            layers.append(QDense(n, beta0=1e-5))

            #quantized batch normalization layer
            layers.append(QBatchNormalization())

            #ReLU activation function
            layers.append(ReLU())

        #Wraps the layers into a Sequential model.
        self.model = Sequential(layers)

    #Passes input through the network
    def forward(self, x):
        #outputs feature embeddings Y and Y'
        return self.model(x)

#Defines the projection head, which maps feature representations to another space.
class ModelProjector(nn.Module):
    def __init__(self, projection_size):
        super().__init__()

        self.projection_size = projection_size

        self.blocks = nn.ModuleList()

        #determines input size, adds batch normalization and ReLU activations - flexible projection head
        for i in range(2):
            self.blocks.append(nn.Sequential(nn.LazyLinear(projection_size), nn.BatchNorm1d(projection_size), nn.ReLU(True)))

        self.blocks.append(nn.Sequential(nn.LazyLinear(projection_size), nn.BatchNorm1d(projection_size)))

    #Passes input through projection layers sequentially (maps Y to Z)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
