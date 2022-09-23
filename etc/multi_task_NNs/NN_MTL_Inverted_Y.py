import torch
import torch.nn as nn

# Define NN
class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()
  
  def forward(self, inp_list):
    result = torch.ones(inp_list[0].size())
    for x in inp_list:
      result *= x
    return result

class NeuralNetwork_MTL(nn.Module): # multi-task learning
  def __init__(self, INPUT_dim, OUTPUT1_dim, OUTPUT2_dim, WIDTH):
    super(NeuralNetwork_MTL, self).__init__()
    self.fc1 = nn.Linear(INPUT_dim, WIDTH, bias=True)
    self.relu1 = nn.ReLU()
    self.mult1 = Multiply()
    self.mult2 = Multiply()
    self.fc2 = nn.Linear(WIDTH, WIDTH, bias=True)
    self.relu2 = nn.ReLU()

    self.mult3 = Multiply()
    self.fc3 = nn.Linear(WIDTH, WIDTH, bias=True)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(WIDTH, OUTPUT1_dim, bias=True)
    
    self.fc5 = nn.Linear(WIDTH, WIDTH, bias=True)
    self.relu4 = nn.ReLU()
    self.fc6 = nn.Linear(WIDTH, OUTPUT2_dim, bias=True)

    nn.init.xavier_normal_(self.fc1.weight)
    nn.init.xavier_normal_(self.fc2.weight)
    nn.init.xavier_normal_(self.fc3.weight)
    nn.init.xavier_normal_(self.fc4.weight)
    nn.init.xavier_normal_(self.fc5.weight)
    nn.init.xavier_normal_(self.fc6.weight)

  def forward(self, x):    
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.mult1([x, x])
    x = self.mult2([x, x])
    x = self.fc2(x)
    x = self.relu2(x)
    
    x1    = self.mult3([x,x])
    x1    = self.fc3(x1)
    x1    = self.relu3(x1)
    value = self.fc4(x1)
      
    x2   = self.fc5(x)
    x2   = self.relu4(x2)
    grad = self.fc6(x2)
    
    return value, grad