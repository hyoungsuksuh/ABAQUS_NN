# ==============================================================
# Step 5: generate ABAQUS .inp file for verification tests
# written by: Chulmin Kweon & Hyoung Suk Suh (Columbia Univ.)
# ==============================================================

# Import necessary packages and functions
import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import os
 
from NN_separate import Multiply, NeuralNetwork_f, NeuralNetwork_dfdsig

# INPUT --------------------------------------------------------
file_name = "Tresca" # Tresca; Hosford; vonMises
# --------------------------------------------------------------




def ExportModel2Txt(file_name):
  if (file_name.lower() == "tresca"):
    model_name_f = "./Tresca/f_NN.pth"
    model_name_df = "./Tresca/dfdsig_NN.pth"

    isExist = os.path.exists("../ABAQUS_UMAT/Tresca")
    if not isExist:
      os.makedirs("../ABAQUS_UMAT/Tresca")

    isExist = os.path.exists("../ABAQUS_UMAT/Tresca/NN_input")
    if not isExist:
      os.makedirs("../ABAQUS_UMAT/Tresca/NN_input")

    output_name_f  = "../ABAQUS_UMAT/Tresca/NN_input/NN_params_f.txt"
    output_name_df = "../ABAQUS_UMAT/Tresca/NN_input/NN_params_df.txt"

  elif (file_name.lower() == "vonmises"):
    model_name_f = "./vonMises/f_NN.pth"
    model_name_df = "./vonMises/dfdsig_NN.pth"

    isExist = os.path.exists("../ABAQUS_UMAT/vonMises")
    if not isExist:
      os.makedirs("../ABAQUS_UMAT/vonMises")

    isExist = os.path.exists("../ABAQUS_UMAT/vonMises/NN_input")
    if not isExist:
      os.makedirs("../ABAQUS_UMAT/vonMises/NN_input")

    output_name_f  = "../ABAQUS_UMAT/vonMises/NN_input/NN_params_f.txt"
    output_name_df = "../ABAQUS_UMAT/vonMises/NN_input/NN_params_df.txt"

  elif (file_name.lower() == "hosford"):
    model_name_f = "./Hosford/f_NN.pth"
    model_name_df = "./Hosford/dfdsig_NN.pth"

    isExist = os.path.exists("../ABAQUS_UMAT/Hosford")
    if not isExist:
      os.makedirs("../ABAQUS_UMAT/Hosford")

    isExist = os.path.exists("../ABAQUS_UMAT/Hosford/NN_input")
    if not isExist:
      os.makedirs("../ABAQUS_UMAT/Hosford/NN_input")

    output_name_f  = "../ABAQUS_UMAT/Hosford/NN_input/NN_params_f.txt"
    output_name_df = "../ABAQUS_UMAT/Hosford/NN_input/NN_params_df.txt"

  else:
    print("\nAvailable file_name")
    print("\tTresca")
    print("\tvonMises\n")
    print("\tHosford\n")

    print("====== Pytorch2Fortran.py is terminated ======")
    exit(1)
 
  print("--------------------------------------------------------------")
  print("Input name1:\n\t",model_name_f)
  print("Input name2:\n\t",model_name_df)
  print("--------------------------------------------------------------")

  model = torch.load(model_name_f)

  weight_width =[]
  for layer in model.children():
    if isinstance(layer, nn.Linear):
      tmp  = layer.state_dict()['weight'].numpy()
      weight_width.append( [len(tmp[0,:]),len(tmp[:,0])]  )

  layer_name = [n for n, _ in model.named_children()]
  layer_width = [str(weight_width[0][0])]
  translated_name = ["input"]
  count = 0 
  for layer in layer_name:
    if "fc" in layer.lower():
      translated_name.append("dense")
      layer_width.append(str(weight_width[count][1]))
      count = count + 1 
    elif "relu" in layer.lower():
      translated_name.append("relu")
      layer_width.append('0')
    elif "sig" in layer.lower():
      translated_name.append("sigmoid")
      layer_width.append('0')
    elif "mult" in layer.lower():
      translated_name.append("multiply")
      layer_width.append('0')
      translated_name.append("linear")
      layer_width.append('0')
    else:
      print("Check the name of layer in the class that defines Neural network" )
      print("Dense\tshould contain 'fc'" )
      print("ReLU\tshould contain 'relu'" )
      print("Sigmoid\tshould contain 'sig'" )
      print("Multiply should contain 'mult'" )
      print("\nIf you have encountered this error, you should change the NN information in .txt manually\n")
      input("===== paused =====")
      
  translated_name.append("linear")    
  layer_width.append('0')

  file_write = open(output_name_f,'w')
  file_write.write(str( len(translated_name) )+"\n")
  for i in range(len(translated_name)):
    file_write.write( translated_name[i]+"\t"+layer_width[i]+"\n")
  for layer in model.children():
    if isinstance(layer, nn.Linear):
      tmp  = layer.state_dict()['bias'].numpy()
      for i in range(len(tmp)):
        file_write.write(str(tmp[i])+"\t")
      file_write.write("\n")

  for layer in model.children():
    if isinstance(layer, nn.Linear):
      tmp  = layer.state_dict()['weight'].numpy()
      for i in range(len(tmp[:,0])):
        for j in range(len(tmp[0,:])):
          file_write.write(str(tmp[i][j])+"\t")
      file_write.write("\n")
  file_write.close

  print("--------------------------------------------------------------")
  print("Output for f is genrated as:\n\t",output_name_f)

  model = torch.load(model_name_df)

  weight_width =[]
  for layer in model.children():
    if isinstance(layer, nn.Linear):
      tmp  = layer.state_dict()['weight'].numpy()
      weight_width.append( [len(tmp[0,:]),len(tmp[:,0])]  )
      
  
  layer_name = [n for n, _ in model.named_children()]
  layer_width = [str(weight_width[0][0])]
  translated_name = ["input"]
  count = 0 
  for layer in layer_name:
    if "fc" in layer.lower():
      translated_name.append("dense")
      layer_width.append(str(weight_width[count][1]))
      count = count + 1 
    elif "relu" in layer.lower():
      translated_name.append("relu")
      layer_width.append('0')
    elif "sig" in layer.lower():
      translated_name.append("sigmoid")
      layer_width.append('0')
    elif "mult" in layer.lower():
      translated_name.append("multiply")
      layer_width.append('0')
      translated_name.append("linear")
      layer_width.append('0')
    else:
      print("Check the name of the layer in the class that defines Neural network" )
      print("Dense\tshould contain 'fc'" )
      print("ReLU\tshould contain 'relu'" )
      print("Sigmoid\tshould contain 'sig'" )
      print("Multiply should contain 'mult'" )
      print("\nIf you have encountered this error, you should change the NN information in .txt manually\n")
      input("===== paused =====")
      
  translated_name.append("linear")    
  layer_width.append('0')

  file_write = open(output_name_df,'w')
  file_write.write(str( len(translated_name) )+"\n")
  for i in range(len(translated_name)):
    file_write.write( translated_name[i]+"\t"+layer_width[i]+"\n")
  for layer in model.children():
    if isinstance(layer, nn.Linear):
      tmp  = layer.state_dict()['bias'].numpy()
      for i in range(len(tmp)):
        file_write.write(str(tmp[i])+"\t")
      file_write.write("\n")

  for layer in model.children():
    if isinstance(layer, nn.Linear):
      tmp  = layer.state_dict()['weight'].numpy()
      
      for i in range(len(tmp[:,0])):
        for j in range(len(tmp[0,:])):
          file_write.write(str(tmp[i][j])+"\t")
      file_write.write("\n")
  file_write.close
  print("Output for df is genrated as:\n\t",output_name_df)
  print("--------------------------------------------------------------\n")
  
  


def GenerateAbaqusInputMaterial(model_name, E = 200000, nu =0.3):
  if (model_name.lower() == "tresca"):
    model_name = "Tresca"
  elif (model_name.lower() == "vonmises"):
    model_name = "vonMises"
  elif (model_name.lower() == "hosford"):
    model_name = "Hosford"
  else:
    print("\nAvailable file_name")
    print("\tTresca")
    print("\tvonMises\n")
    print("\tHosford\n")

    print("====== Pytorch2Fortran.py is terminated ======")
    exit(1)

  input_name = "/NN_input/NN_params"
  input_name1 = "../ABAQUS_UMAT/" + model_name + input_name + "_f.txt"

  if not os.path.exists(input_name1):
    print("\nCheck the name of NN parameter file")
    print("\t"+input_name1)
    print("\tThere is no such file")
    exit(1)
    
  file_read  = open(input_name1,'r')
  line   = file_read.readline()
  Nlines = line
  for i in range(int(Nlines)):
    line = file_read.readline().strip('\n')
    tmp = line .split('\t')
    
  num_para = 1
  while line:  
    line = file_read.readline().strip('\n')
    tmp = line .split('\t')
    for param in tmp:
      if len(param) != 0:
        num_para = num_para +1
  file_read.close

  input_name2 = "../ABAQUS_UMAT/" + model_name + input_name+ "_df.txt"
  
  if not os.path.exists(input_name2):
    print("\nCheck the name of the NN parameter file")
    print("\t"+input_name2)
    print("\tThere is no such file")
    exit(1)
    
  file_read   = open(input_name2,'r')
  line        = file_read.readline()
  Nlines = line
  for i in range(int(Nlines)):
    line = file_read.readline().strip('\n')
    tmp  = line .split('\t')
    
  while line:  
    line = file_read.readline().strip('\n')
    tmp  = line .split('\t')
    for param in tmp:
      if len(param) != 0:
        num_para = num_para +1
  file_read.close

  file_read  = open(input_name1,'r')
  file_write = open("../ABAQUS_UMAT/" + model_name + "/NN_input/MaterialParameter.txt",'w')
  line   = file_read.readline()
  Nlines = line
  for i in range(int(Nlines)):
    line = file_read.readline().strip('\n')
    
  file_write.write('**\n')
  file_write.write('** MATERIALS\n')
  file_write.write('**\n')
  file_write.write('*Material, name=Material-1\n')
  file_write.write('*Depvar\n     13,\n')
  file_write.write('1, EE11, EE11\n')
  file_write.write('2, EE22, EE22\n')
  file_write.write('3, EE33, EE33\n')
  file_write.write('4, EE12, EE12\n')
  file_write.write('5, EE13, EE13\n')
  file_write.write('6, EE23, EE23\n')
  file_write.write('7, EP11, EP11\n')
  file_write.write('8, EP22, EP22\n')
  file_write.write('9, EP33, EP33\n')
  file_write.write('10, EP12, EP12\n')
  file_write.write('11, EP13, EP13\n')
  file_write.write('12, EP23, EP23\n')
  file_write.write('13, PEEQ, PEEQ\n')
  converted_num = "%s" % (num_para+1)
  file_write.write('*User Material, constants='+converted_num+'\n')

  tmp_str = str(E)+", "+str(nu)+", "
  file_write.write(tmp_str)
  line_ind = 2
  while line:  
    line = file_read.readline().strip('\n')
    tmp = line .split('\t')
    for param in tmp:
      if len(param) != 0:
        line_ind = line_ind +1
        file_write.write(param)
        if( line_ind%8 == 0):
          file_write.write(',\n')
        else:
          file_write.write(', ')

  file_read.close
  file_write.close

  file_read  = open(input_name2,'r')
  line   = file_read.readline()
  Nlines = line
  for i in range(int(Nlines)):
    line = file_read.readline().strip('\n')
    
  while line:  
    line = file_read.readline().strip('\n')
    tmp = line .split('\t')
    for param in tmp:
      if len(param) != 0:
        line_ind = line_ind +1
        file_write.write(param)
        if( line_ind%8 == 0):
          file_write.write(',\n')
          
        else:
          file_write.write(', ')
    
  file_read.close
  file_write.close
  print("Material Parameter file is generated\n\t ../ABAQUS_UMAT/" + model_name + "/NN_input/MaterialParameter.txt")
  print("\tTotal number of parameter is:",line_ind)
  



def GenerateVerificationInput(model_name, BVP):
  if (model_name.lower() == "tresca"):
    model_name = "Tresca"
  elif (model_name.lower() == "vonmises"):
    model_name = "vonMises"
  elif (model_name.lower() == "hosford"):
    model_name = "Hosford"
  else:
    print("\nAvailable file_name")
    print("\tTresca")
    print("\tvonMises\n")
    print("\tHosford\n")
    print("====== Pytorch2Fortran.py is terminated ======")
    exit(1)

  if (BVP.lower() == "planar_tension"):
    BVP = "planar_tension"
  elif (BVP.lower() == "uniaxial_tension"):
    BVP = "uniaxial_tension"
  elif (BVP.lower() == "biaxial_tension"):
    BVP = "biaxial_tension"
  elif (BVP.lower() == "simple_shear"):
    BVP = "simple_shear"
  else:
    print("\nAvailable boundary value problem :")
    print("\tuniaxial_tension\n")
    print("\tbiaxial_tension\n")
    print("\tsimple_shear\n")
    print("\tpure_shear")
    print("====== Pytorch2Fortran.py is terminated ======")
    exit(1)

  file_write = open("../ABAQUS_UMAT/" + model_name + "/"+ model_name+"_"+BVP+".inp",'w')
  file_read1  = open("./util/AbaqusInpFormat/" + BVP + ".inp",'r')
  file_read2  = open("../ABAQUS_UMAT/" + model_name + "/NN_input/MaterialParameter.txt",'r')

  line1 = file_read1.readline().strip('\n')
  tmp_str =  line1+"\n"
  file_write.write(tmp_str)
  file_write.write("** Job name: " + model_name + "_" + BVP + " name: Model-1\n")
  line1 = file_read1.readline().strip('\n')
  line1 = file_read1.readline().strip('\n')

  while line1:
    tmp_str =  line1+"\n"
    file_write.write(tmp_str)
    line1 = file_read1.readline().strip('\n')
    if line1 == "** BOUNDARY CONDITIONS":
      break

  line2 = file_read2.readline().strip('\n')
  while line2:
    tmp_str =  line2+"\n"
    file_write.write(tmp_str)
    line2 = file_read2.readline().strip('\n')

  while line1:
    tmp_str =  line1+"\n"
    file_write.write(tmp_str )
    line1 = file_read1.readline().strip('\n')
  file_write.close
  file_read1.close
  file_read2.close

  print("\nAbaqus input file for ",file_name + " - " + verification + " is generated")
  print("\t../ABAQUS_UMAT/" + model_name + "/"+ model_name+"_"+BVP+".inp")




BVP = ["uniaxial_tension", "biaxial_tension", "simple_shear", "planar_tension"]
ExportModel2Txt(file_name)
GenerateAbaqusInputMaterial(file_name)
for verification in BVP:
  GenerateVerificationInput(file_name, verification)