# ==============================================================
# Step 3: train NN - plastic flow direction
# written by: Hyoung Suk Suh & Chulmin Kweon (Columbia Univ.)
# ==============================================================

# Import necessary packages and functions
import os
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import autograd.numpy as np

from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from autograd import elementwise_grad as egrad

from util.tensor_operations import *
from util.coordinate_transforms import *

from NN_separate import Multiply, NeuralNetwork_dfdsig

# INPUT --------------------------------------------------------
# NN parameters (including hyperparameters)
WIDTH = 100   # Hidden layer width
BATCH = 128   # Batch size
EPOCH = 2000  # Training epochs
LR    = 0.001 # Learning rate
SPLIT = 0.2   # train : validation = 1-SPLIT : SPLIT

# File name
file_name = "Tresca"
# --------------------------------------------------------------




# Load training dataset
data = pd.read_csv("./"+file_name+"/training_data.csv")

# >> Input dataset
p_level_set     = data["p"].values
rho_level_set   = data["rho"].values
theta_level_set = data["theta"].values
lamda_level_set = data["lamda"].values

p_level_set     = p_level_set.reshape(-1,1)
rho_level_set   = rho_level_set.reshape(-1,1)
theta_level_set = theta_level_set.reshape(-1,1)
lamda_level_set = lamda_level_set.reshape(-1,1)

# >> Dataset for training dgdsig
dfdsig1_level_set = data["dfdsig1"].values
dfdsig2_level_set = data["dfdsig2"].values
dfdsig3_level_set = data["dfdsig3"].values

dfdsig1_level_set = dfdsig1_level_set.reshape(-1,1)
dfdsig2_level_set = dfdsig2_level_set.reshape(-1,1)
dfdsig3_level_set = dfdsig3_level_set.reshape(-1,1)




# Define NN input
min_p     = np.min(p_level_set)
min_rho   = np.min(rho_level_set)
min_theta = np.min(theta_level_set)
min_lamda = np.min(lamda_level_set)

max_p     = np.max(p_level_set)
max_rho   = np.max(rho_level_set)
max_theta = np.max(theta_level_set)
max_lamda = np.max(lamda_level_set)

INPUT = np.concatenate((p_level_set, rho_level_set, theta_level_set, lamda_level_set), axis=1)
INPUT_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
INPUT_scaled = INPUT_scaler.fit_transform(INPUT)
joblib.dump(INPUT_scaler, "./"+file_name+"/dfdsig_INPUT_scaler.pkl")

INPUT_scaler_txt = np.hstack([min_p, max_p, min_rho, max_rho, min_theta, max_theta, min_lamda, max_lamda])
np.savetxt("../ABAQUS_UMAT/"+file_name+"/NN_input/dfins.txt", INPUT_scaler_txt)




# Define NN outputs
# >> dfdsig
OUTPUT = np.concatenate((dfdsig1_level_set, dfdsig2_level_set, dfdsig3_level_set), axis=1)
OUTPUT_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
OUTPUT_scaled = OUTPUT_scaler.fit_transform(OUTPUT)
joblib.dump(OUTPUT_scaler, "./"+file_name+"/dfdsig_OUTPUT_scaler.pkl")

min_dfdsig1 = np.min(dfdsig1_level_set)
min_dfdsig2 = np.min(dfdsig2_level_set)
min_dfdsig3 = np.min(dfdsig3_level_set)
max_dfdsig1 = np.max(dfdsig1_level_set)
max_dfdsig2 = np.max(dfdsig2_level_set)
max_dfdsig3 = np.max(dfdsig3_level_set)
OUTPUT_scaler_txt = np.hstack([min_dfdsig1, max_dfdsig1, min_dfdsig2, max_dfdsig2, min_dfdsig3, max_dfdsig3])
np.savetxt("../ABAQUS_UMAT/"+file_name+"/NN_input/dfouts.txt", OUTPUT_scaler_txt)




# Define trainer
class L2_trainer():

  # Initialize trainer
  def __init__(self, sINPUT, sOUTPUT, INPUT_scaler, OUTPUT_scaler, SPLIT, WIDTH, EPOCH, BATCH, LR):

    # Hyperparameters
    self.WIDTH = WIDTH
    self.EPOCH = EPOCH
    self.BATCH = BATCH
    self.LR    = LR

    # Number of data (no. of training & validation sets)
    self.N_data  = int(sINPUT.shape[0])
    self.N_train = int((1-SPLIT)*self.N_data)
    self.N_valid = self.N_data - self.N_train

    # Input & output dimensions
    self.INPUT_dim   = int(sINPUT.shape[1])
    self.OUTPUT_dim = int(sOUTPUT.shape[1])

    # Scalers
    self.INPUT_scaler  = INPUT_scaler
    self.OUTPUT_scaler = OUTPUT_scaler

    self.INPUT_scales  = torch.tensor(self.INPUT_scaler.scale_, dtype=torch.float)
    self.OUTPUT_scales = torch.tensor(self.OUTPUT_scaler.scale_, dtype=torch.float)

    self.sINPUT  = np.copy(sINPUT)
    self.sOUTPUT = np.copy(sOUTPUT)

    # Training history
    self.history = {'train_loss':[], 'valid_loss':[]}

    # Initialize losses
    self.train_loss = 0.
    self.valid_loss = 0.

    # Define model
    self.device = torch.device("cpu")
    self.model = NeuralNetwork_dfdsig(self.INPUT_dim, self.OUTPUT_dim, self.WIDTH).to(self.device)
    self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.LR)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.95, threshold=1e-6, min_lr=1e-16)


  def DataShuffle(self):
    # Shuffle all samples
    s = np.arange(self.sINPUT.shape[0])
    np.random.shuffle(s)

    # Split samples into training and validation sets
    sINPUT = np.copy(self.sINPUT[s])
    sINPUT = torch.tensor(sINPUT, dtype=torch.float)
    sINPUT = torch.unsqueeze(sINPUT, 1)

    sOUTPUT = np.copy(self.sOUTPUT[s])
    sOUTPUT = torch.tensor(sOUTPUT, dtype=torch.float)
    sOUTPUT = torch.unsqueeze(sOUTPUT, 1)

    sINPUT_train  = sINPUT[0:self.N_train, :]
    sOUTPUT_train = sOUTPUT[0:self.N_train, :]

    sINPUT_valid  = sINPUT[self.N_train:, :]
    sOUTPUT_valid = sOUTPUT[self.N_train:, :]

    train_dataset = TensorDataset(sINPUT_train, sOUTPUT_train)
    valid_dataset = TensorDataset(sINPUT_valid, sOUTPUT_valid)

    self.train_loader = DataLoader(train_dataset, batch_size=self.BATCH, shuffle=True)
    self.valid_loader = DataLoader(valid_dataset, batch_size=self.BATCH, shuffle=True)


  # Loss function
  def loss_fun(self, pred, true):
    return torch.mean((pred - true)**2)


  # Training loop
  def train_loop(self):

    for batch_idx, samples in enumerate(self.train_loader):

      sINPUT_train, sOUTPUT_train = samples
      sINPUT_train.requires_grad = True

      sOUTPUT_train_predicted = self.model(sINPUT_train)

      # Compute loss
      train_loss_batch = self.loss_fun(sOUTPUT_train_predicted, sOUTPUT_train)

      self.train_loss += train_loss_batch.detach().numpy() * len(sOUTPUT_train)

      # Update weights & biases
      self.optimizer.zero_grad()
      train_loss_batch.backward()
      self.optimizer.step()

    # Record training history
    self.train_loss /= self.N_train
    self.history['train_loss'].append(self.train_loss)

    # Update LR if its stuck
    self.scheduler.step(self.train_loss)


  # Validation loop
  def test_loop(self):

    for batch_idx, samples in enumerate(self.valid_loader):

      sINPUT_valid, sOUTPUT_valid = samples

      # Prediction
      sOUTPUT_valid_predicted = self.model(sINPUT_valid)

      # Compute loss
      valid_loss_batch = self.loss_fun(sOUTPUT_valid_predicted, sOUTPUT_valid)

      self.valid_loss += valid_loss_batch.detach().numpy() * len(sOUTPUT_valid)

    # Record validation history
    self.valid_loss /= self.N_valid
    self.history['valid_loss'].append(self.valid_loss)


  # Print history
  def print_history(self, t):
    if t % 1 == 0:
      print(t, \
            ">>","Training loss:","{:.3e}".format(self.train_loss.item()), \
            ";","Validation loss:","{:.3e}".format(self.valid_loss.item()), \
            ";","LR:","{:.3e}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
      print("-------------------------------------------------------------------------------------")


  # Run training
  def run(self):
    self.DataShuffle()
    for t in range(self.EPOCH):
      self.train_loss = 0
      self.valid_loss = 0

      self.train_loop()
      self.test_loop()
      self.print_history(t)
      self.save(t)


  # Save
  def save(self, t):
    if t % 10 == 0:
      torch.save(self.model, "./"+file_name+"/dfdsig_NN.pth")

    if t % 10 == 0:
      output_history = pd.DataFrame({'train_loss': self.history['train_loss'], \
                                     'valid_loss': self.history['valid_loss']})

      output_history.to_csv("./"+file_name+"/dfdsig_training_history.csv", index = False)




# Train NN
print("=====================================================================================")
print("Level set plasticity: stress gradient")
print(">> Written by: Hyoung Suk Suh and Chulmin Kweon")
print("=====================================================================================")
trainer = L2_trainer(INPUT_scaled, OUTPUT_scaled, INPUT_scaler, OUTPUT_scaler, \
                     SPLIT, WIDTH, EPOCH, BATCH, LR)
trainer.run()




# Plot history
train_loss_history = trainer.history['train_loss']
valid_loss_history = trainer.history['valid_loss']

plt.figure(0,figsize=(7,7))
plt.semilogy(train_loss_history, 'b-', linewidth=1.0, label="Training loss")
plt.semilogy(valid_loss_history, 'r--', linewidth=1.0, label="Validation loss")
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(loc="upper right")
plt.show()
