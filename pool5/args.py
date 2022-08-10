import torch

max_epoch =20
hidden_dim = 7*7*4
batch_size = 2
reg_para = 0.25
train_size = 0.5
learning_rate = 0.01
outer_learning_rate =0.04





device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
datafile = "/raid/zy_lab/zhangyi/code_NGTTN/data/office_caltech_pool5.txt"

#datafile = "../data/office_caltech_fc7.txt"










