import torch

max_epoch =100
hidden_dim = 128
batch_size = 2
reg_para = 0.25
train_size = 0.5
learning_rate = 0.01
outer_learning_rate =0.04

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
datafile = "../data/office_caltech_fc7.txt"








