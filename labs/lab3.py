import os
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Callable
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import LedoitWolf
from utils.lab_03_utils import *

###########################################################################
###########################################################################
### 1.1 Binary classifier with Mahalanobis distance [2.5 pts]

class MahalanobisClassifier:
    """Mahalanobis based classifer"""

    def __init__(self):
        """
        Attributes:
            means (torch.tensor): (n_classes, d) Mean of the features for each class
            inv_covs (torch.tensor): (n_classes, d, d) Inverse of covariance matrix across d features for each class
        """
        super().__init__()
        self.means = None
        self.inv_covs = None

    def fit(self, train_x : torch.Tensor, train_y : torch.Tensor):
        """Computes parameters for Mahalanobis Classifier (self.mean and self.cov), fitted on the training data.

        Args:
            train_x (torch.Tensor): (N, d) The tensor of training features
            train_y (torch.Tensor): (N,) The tensor of training labels
        """

        # Define number of classes
        n_classes = len(np.unique(np.unique(train_y)))
        n, d = train_x.shape

        # Set default values
        means = torch.zeros((n_classes, d), dtype=train_x.dtype)
        inv_covs = torch.ones((n_classes, d, d), dtype=train_x.dtype)

        # ------------------
        # Your code here ...
        # ------------------
        # There's 2 classes in the dataset, so we can compute the mean and covariance on the whole dataset for each class.
        # Mean size is (2, d)
        for i in range(n_classes):
            class_x = train_x[train_y == i]
            means[i] = torch.mean(class_x, dim=0)
            cov = torch.cov(class_x.T)
            inv_covs[i] = torch.linalg.inv(cov)

        self.means = means
        self.inv_covs = inv_covs



    def predict(self, test_x : torch.Tensor) -> torch.Tensor:
        """Predicts the class of every test feature, using the Mahalanobis Distance

        Args:
            test_x (torch.Tensor): (N, d) The tensor of test features

        Returns:
            preds (torch.Tensor): (N,) The predictions tensor (id of the predicted class {0, 1, ..., n_classes-1})
            dists (torch.Tensor): (N, n_classes) Mahalanobis distance from sample to class means
        """

        # Define default output value
        N, d = test_x.shape
        dists = torch.zeros((N, self.means.shape[0]), dtype=test_x.dtype)
        preds = torch.zeros(N, dtype=test_x.dtype)

        # ------------------
        # Your code here ...
        # ------------------
        # Compute the Mahalanobis distance from each test sample to each class mean, using the inverse covariance matrix.
        for i in range(self.means.shape[0]):
            diff = test_x - self.means[i]
            dists[:, i] = torch.sqrt(torch.sum(diff @ self.inv_covs[i] * diff, dim=1))
            # if dists is Nan, it means that the covariance matrix is not invertible, which can happen if the features are not linearly independent. In this case, we can add a small value to the diagonal of the covariance matrix to make it invertible.

        # Predict the class with the smallest Mahalanobis distance
        preds = torch.argmin(dists, dim=1)

        return preds, dists
    




###########################################################################
###########################################################################
### 1.2 Out-of-Distribution detection with Mahalanobis distance [3.5 pts]


class MahalanobisOODClassifier(MahalanobisClassifier):
    """Predicts the class of every test feature, using the Mahalanobis Distance

    Args:
        test_x (torch.Tensor): (N x d) The tensor of test features

    Returns:
        preds (torch.Tensor): (N,) The predictions tensor (id of the predicted class {0, 1, ..., n_classes-1})
        dists (torch.Tensor): (N, n_classes) Mahalanobis distance from sample to class means
        ood_scores (torch.Tensor): (N,) Score of OoDness as the minimal distance from the sample to classes
    """

    def predict(self, test_x : torch.Tensor) -> torch.Tensor:

        # Get super prediction (from MahalanobisClassifier)
        preds, dists = super().predict(test_x=test_x)
        N = preds.shape[0]

        # Assign dummy values to scores
        ood_scores = np.zeros(N)

        # ------------------
        # Your code here ...
        # ------------------
        print("dists: ", dists)

        ood_scores = torch.min(dists, dim=1).values
        print("ood scores: ", ood_scores)
        return preds, dists, ood_scores


def get_ood_threshold(ood_scores, quantile=0.95):
    """ Get OoD threshold based on measured scores and quantile

    Args:
        ood_scores (torch.Tensor): (N, ) N measured OoDness scores
        quantile (float): Percentage of samples that are considered as in distribution
    """

    # Set default value
    # 95% is 2 sigma for a normal distribution, so we can use it as a default threshold.
    threshold = 0

    # ------------------
    # Your code here ...
    # ------------------
    print("ood scores: ", ood_scores)
    # cov = torch.cov(ood_scores.T)
    # threshold = 2*torch.sqrt(cov) # We can use the standard deviation of the scores to set the threshold, so that we consider as OoD the samples that are more than 2 standard deviations away from the mean.
    threshold = torch.quantile(ood_scores[~torch.isnan(ood_scores)], quantile) # What does this function do? It returns the value below which a given percentage of data falls. So if we set quantile to 0.95, it will return the value below which 95% of the data falls, which means that 5% of the data will be above this threshold and considered as OoD.
    # When it returns nan what does it means ?
    return threshold


def compute_metrics(y, y_hat, ood_scores, threshold):
    """ Compute recall for tumor, stroma, and OoD as well as the average recall.

    Args:
        y (torch.Tensor): (N) Class ground truth {-1, 0, 1, ..., n_classes}
        y_hat (torch.Tensor): (N,) Class predictions {0, 1, ..., n_classes}
        ood_scores (torch.Tensor): (N, ) N measured OoDness scores
        threshold (float): OoD threshold
    """
    # Define variable with dummy values
    recall_tumor = 0
    recall_stroma = 0
    recall_ood = 0
    avg_recall = 0

    # ------------------
    # Your code here ...
    # ------------------

    #Complete the function `compute_metrics` that computes the recall for TUMOR, STROMA, and OoD examples as well as the average recall over the 3 classes. To do so, you need to consider OoDs as a third class by assigning the prediction `-1` to filtered-out examples based on your threshold.
    y_hat_ood = copy.deepcopy(y_hat)
    y_hat_ood[ood_scores > threshold] = -1 # We assign the prediction -1 to the samples that are considered as OoD based on the threshold.
    recall_tumor = torch.sum((y == 0) & (y_hat_ood == 0)) / torch.sum(y == 0)
    recall_stroma = torch.sum((y == 1) & (y_hat_ood == 1)) / torch.sum(y == 1)
    recall_ood = torch.sum((y == -1) & (y_hat_ood == -1)) / torch.sum(y == -1)
    avg_recall = (recall_tumor + recall_stroma + recall_ood) / 3

    return recall_tumor, recall_stroma, recall_ood, avg_recall

    


    

###########################################################################
###########################################################################    
### 1.3 Out-of-Distribution detection with k-NN [5 pts]

class kNNClassifier:
    """k-NN based classifier"""

    def __init__(self, k : int):
        """
        Args:
            k (int): The number of neighbors to consider for the classification
            features (torch.Tensor): (N, d) feature of the N train samples
            labels (torch.Tensor): (N,) labels for train samples
        """
        self.k = k
        self.features = None
        self.labels = None

    def fit(self, train_x : torch.Tensor, train_y : torch.Tensor):
        """Store training data parameters (features and labels) for k-NN classifier.

        Args:
            train_x (torch.Tensor): (N, d) The tensor of training features
            train_y (torch.Tensor): (N,) The tensor of training labels
        """

        # Get size and default values
        N, d = train_x.shape
        features = torch.zeros((N, d))
        labels = torch.zeros(N)

        # ------------------
        # Your code here ...
        # ------------------
        features = train_x
        labels = train_y

        self.features = features
        self.labels = labels

    def predict(self, test_x : torch.Tensor) -> torch.Tensor:
        """Predicts the class of every test feature, using the k-NN

        Args:
            test_x (torch.Tensor): (N x d) The tensor of test features

        Returns:
            preds (torch.Tensor): (N,) The tensor of class predictions {0, 1, ..., n_classes}
            ood_scores (torch.Tensor): (N,) The OoD score predictions
        """


        # Get size and default values
        N, d = test_x.shape
        preds = torch.zeros(N)
        ood_scores = torch.zeros(N)

        # ------------------
        # Your code here ...
        # ------------------
        for i in range(N):
            # Compute the distance from the test sample to all training samples
            distances = torch.sqrt(torch.sum((self.features - test_x[i])**2, dim=1))
            # Get the indices of the k nearest neighbors
            knn_indices = torch.argsort(distances)[:self.k]
            # Get the labels of the k nearest neighbors
            knn_labels = self.labels[knn_indices]
            # Predict the class as the majority class among the k nearest neighbors
            preds[i] = torch.mode(knn_labels).values
            # The OoD score can be defined as the distance to the nearest neighbor, so we can use the distance to the closest training sample as the OoD score.
            ood_scores[i] = distances[knn_indices[0]]

        return preds, ood_scores
    
# Best k for knn fitting (to find among suggested ks)

def find_best_k(ks,kNNClassifier: Callable,train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor):
    best_k = 0
    best_accuracy = 0.
    # Iterate over ks
    for k in ks:

        # ------------------
        # Your code here ...
        # ------------------
        # For each k, fit a kNN classifier on the training data and evaluate its accuracy on the validation set. Return the best k and the corresponding accuracy.
        classifier = kNNClassifier(k)
        classifier.fit(train_x, train_y)
        preds, _ = classifier.predict(val_x)
        accuracy = accuracy_score(val_y, preds)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

        continue

    return best_k, best_accuracy

def fit_knn(best_k, train_x, train_y, val_x, val_y):

    # best threshold
    threshold_val = 0
    # Predicted val ood scores
    val_y_ood_scores = torch.zeros(len(val_y))
    classifier = None
    # ------------------
    # Your code here ...
    # ------------------
    # Fit a kNN classifier with the best k found on the training data, and compute the OoD scores on the validation set. Then, find the best threshold for OoD detection based on the validation set.
    classifier = kNNClassifier(best_k)
    classifier.fit(train_x, train_y)
    _, val_y_ood_scores = classifier.predict(val_x)
    threshold_val = get_ood_threshold(val_y_ood_scores)

    return classifier, threshold_val, val_y_ood_scores






######################################################################################################################################################
###################################################################################################################################################### 
            ## Part 2 - Lung Adenocarcinoma Classification (19 points)
######################################################################################################################################################
######################################################################################################################################################
    



###########################################################################
###########################################################################    
### 2.1 Dataset [1 pt]

class DHMC2Cls(Dataset):
    """DHMC dataset using 2 classes"""

    def __init__(self, features_path : str, train : bool = False) -> None:
        """
        Attributes:
            raw_data (list of dict): (M) List of M slides raw data as dictionaries.
            train (bool): True if data are the training set. False otherwise

        Args:
            features_path (str): The path to the features file
            train (bool): Whether it is the training dataset or not
        """

        super().__init__()
        # Load raw data from path
        self.raw_data = torch.load(features_path, weights_only=False)
        # Set if training or not
        self.train = train

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: The length M of the dataset
        """

        n_data = 0

        # ------------------
        # Your code here ...
        # ------------------
        n_data = len(self.raw_data)

        return n_data

    def __getitem__(self, index : int):
        """Returns the entry at index from the dataset

        Args:
            index (int): the requested entry index of the dataset

        Returns:
            features (torch.Tensor): (N, d) Feature tensor of the selected slide with N patches and d feature dimensions
            label (int): Ground truth label {0, ..., n_classes}
            wsi_id (str): Name of the WSI as "DHMC_xxx" where xxx is a unique id of the slide (train == False only)
            coordinates (torch.Tensor): (N, 2) xy coordinates of the N patches of the selected slide (train == False only)
        """

        features = None
        label = None
        wsi_id = None
        coordinates = None

        # ------------------
        # Your code here ...
        # ------------------
        item = self.raw_data[index]
        features = item["patch_features"]
        label = item["label"]

        if self.train:
            return features, label

        wsi_id = item["wsi_id"]
        coordinates = item["patch_coordinates"]
        return features, label, wsi_id, coordinates

    

###########################################################################
###########################################################################    
### 2.2 Average Pooling [1 pt]

class AveragePooling(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features : torch.Tensor):
        """ Perform mean along the first dimension of the tensor

        Args:
            features (torch.Tensor): (N, D) Feature to perform average pooling on
        Return:
            mean (torch.Tensor): (1, D) Features average over all patches
        """

        mean = None

        # ------------------
        # Your code here ...
        # ------------------
        mean = torch.mean(features, dim=0, keepdim=True)

        return mean

    

###########################################################################
###########################################################################    
### 2.3 Classifier [8 pts]

class LinearClassifier(nn.Module):

    def __init__(self, in_dim : int, H : int, n_classes : int, pooling_fn : nn.Module) -> None:
        """Constructs the linear classifier

        Attributes:
            proj (Callable): Projection of layer (N, d) -> (N, H)
            pool (Callable): Pooling layer (N, H) -> (1, H)
            fc (Callable): Classification layer (1, H) -> (1, n_classes)

        Args:
            in_dim (int): The dimension of input features
            H (int): Target dimension for the projection layer
            n_classes (int): The number of classes for the task
            pooling_fn (nn.Module): The pooling function to aggregate the features
        """
        super().__init__()

        proj_layer = None
        pool_layer = None
        fc_layer = None

        # ------------------
        # Your code here ...
        # ------------------
        proj_layer = nn.Linear(in_dim, H)
        pool_layer = pooling_fn
        fc_layer = nn.Linear(H, n_classes)

        self.proj = proj_layer
        self.pool = pool_layer
        self.fc = fc_layer


    def forward(self, x):
        """Forward path

        Args:
            x (torch.Tensor): (1, N, d) Input feature for a given slide with N patches
        Return:
            logits (torch.Tensor): (1, n_classes) Output logits for classification
        """

        logits = None

        # ------------------
        # Your code here ...
        # ------------------
        # x: (1, N, d) -> (N, d)
        x = x.squeeze(0)
        x = self.proj(x)          # (N, H)
        x = F.relu(x)             # required by statement
        x = self.pool(x)          # (1, H)
        logits = self.fc(x)       # (1, n_classes)
        return logits
    
 

def train(model : nn.Module, train_loader : DataLoader, val_loader : DataLoader, n_epochs : int, optimizer : torch.optim.Optimizer):
    """Trains the neural network self.model for n_epochs using a given optimizer on the training dataset.
    Outputs the best model in terms of F1 score on the validation dataset.

    **Notes**: 
    * Refer to this [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for guidance on training a classifier.
    * To obtain the model checkpoint, simply call `model.state_dict()`.
    * We provide you the `test` function, in utils file, that computes the F1 score on a given test dataset. **You should not modify it !!!**

    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): The training dataloader to iterate on the training dataset
        val_loader (DataLoader): The validation dataloader to iterate on the validation dataset
        n_epochs (int): The number of epochs, i.e. the number of time the model should see each training example
        optimizer (torch.optim.Optimizer): The optimizer function to update the model parameters

    Returns:
        best_model (nn.Module): Best model state dictionary
        best_f1 (float): Best F1-score on the validation set
        best_epoch (int): Best epoch on validation set
        val_f1s (list of floats): (n_epochs, ) F1-scores for all epochs
        val_losses (list of floats): (n_epochs, ) Losses for all validation epochs
        train_losses(list of floats): (n_epochs, ) Losses for all training epochs
    """

    # Initialize variable to return
    best_model = model.state_dict()
    best_epoch = 0
    best_f1 = 0
    train_losses = []
    val_losses = []
    val_f1s = []

    # ------------------
    # Your code here ...
    # ------------------
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
            features, labels = batch[:2]
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch[:2]
                logits = model(features)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = f1_score(all_labels, all_preds, average="macro")
        val_f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model.state_dict())

    return best_model, best_f1, best_epoch, val_f1s, val_losses, train_losses
    

###########################################################################
###########################################################################    
### 2.4 Attention Pooling [9 pts]

class Attn_Net_Gated(nn.Module):
    def __init__(self, L : int, M : int):
        """
        Attention Network with Sigmoid Gating (3 fc layers)
        Args:
            L: input feature dimension
            M: hidden layer dimension
        """
        super(Attn_Net_Gated, self).__init__()

        # ------------------
        # Your code here ...
        # ------------------
        self.fc1 = nn.Linear(L, M)
        self.fc2 = nn.Linear(L, M)
        self.fc3 = nn.Linear(M, 1)

    def forward(self, x):
        """Forward path of the gated attention network

        Args:
            xin: (N, L) List of N patches and L features
        Return:
            A: (N, 1) Attention value for each patch
        """
        A = torch.zeros((1,), dtype=x.dtype)
        # ------------------
        # Your code here ...
        # ------------------
        a = torch.tanh(self.fc1(x)) * torch.sigmoid(self.fc2(x))
        A = self.fc3(a)

        return A

class AttentionPooling(nn.Module):
    def __init__(self, L : int, M : int):
        super().__init__()
        # Intatiate the gated layer
        self.attention_net = Attn_Net_Gated(L, M)

    def forward(self, x, attention_only : bool = False):
        """Forward pass

        Args:
            x (torch.tensor): (N, L) Input feature over N patches and L features
            attention_only (bool): Say whether to return the attention or not
        Returns:
            Y (torch.Tensor): (1, N) Output, if attention_only==False
            A (torch.Tensor): (1, M) Attention values, if attention_only==True
        """

        A = None
        Y = None

        # ------------------
        # Your code here ...
        # ------------------
        A = self.attention_net(x) # (N, 1)
        A = torch.transpose(A, 1, 0) # (1, N)
        A = F.softmax(A, dim=1) # (1, N)
        Y = A @ x # (1, L)

        # Check if need to return attention
        if attention_only:
            return A
        else:
            return Y
        
    
###########################################################################
###########################################################################   




    


