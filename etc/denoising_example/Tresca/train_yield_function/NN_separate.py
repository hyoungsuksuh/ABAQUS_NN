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

class NeuralNetwork_f(nn.Module):
  def __init__(self, INPUT_dim, OUTPUT_dim, WIDTH):
    super(NeuralNetwork_f, self).__init__()
    self.fc1 = nn.Linear(INPUT_dim, WIDTH, bias=True)
    self.relu1 = nn.ReLU()
    self.mult1 = Multiply()
    self.mult2 = Multiply()
    self.fc2 = nn.Linear(WIDTH, WIDTH, bias=True)
    self.relu2 = nn.ReLU()
    self.mult3 = Multiply()
    self.fc3 = nn.Linear(WIDTH, WIDTH, bias=True)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(WIDTH, OUTPUT_dim, bias=True)
    
    nn.init.xavier_normal_(self.fc1.weight)
    nn.init.xavier_normal_(self.fc2.weight)
    nn.init.xavier_normal_(self.fc3.weight)
    nn.init.xavier_normal_(self.fc4.weight)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.mult1([x,x])
    x = self.mult2([x,x])
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.mult3([x,x])
    x = self.fc3(x)
    x = self.relu3(x)
    x = self.fc4(x)
    return x
 
class NeuralNetwork_dfdsig(nn.Module):
  def __init__(self, INPUT_dim, OUTPUT_dim, WIDTH):
    super(NeuralNetwork_dfdsig, self).__init__()
    self.fc1 = nn.Linear(INPUT_dim, WIDTH, bias=True)
    self.relu1 = nn.ReLU()
    self.mult1 = Multiply()
    self.mult2 = Multiply()
    self.fc2 = nn.Linear(WIDTH, WIDTH, bias=True)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(WIDTH, WIDTH, bias=True)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(WIDTH, OUTPUT_dim, bias=True)
    
    nn.init.xavier_normal_(self.fc1.weight)
    nn.init.xavier_normal_(self.fc2.weight)
    nn.init.xavier_normal_(self.fc3.weight)
    nn.init.xavier_normal_(self.fc4.weight)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.mult1([x,x])
    x = self.mult2([x,x])
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    x = self.relu3(x)
    x = self.fc4(x)
    return x