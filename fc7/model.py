import torch
import torch.nn as nn
from torch.nn import Parameter
import args as Args
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from torch import diag

class DMTL_net_(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(DMTL_net_, self).__init__()  #
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim) #用self.imput???
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num) )
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)


    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs

    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        # for i in range(self.task_num):
        #     regular_term += torch.pow(torch.norm(self.classifier_layer[i][0].weight), 2)
        regular_term += torch.pow(torch.norm(self.cls_weight), 2)
        return regular_term


class TNRMTL_Tucker_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, class_num, task_num):

        super(TNRMTL_Tucker_net, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num) )
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += Tucker_Norm(self.cls_weight)
        return regular_term
def Tucker_Norm(X):
    shapeX = X.size()
    dimX = len(shapeX)
    re = [nuclear_norm(i) for i in [TensorUnfold(X, j) for j in range(dimX)]]
    return torch.mean(torch.stack(re))

class TNRMTL_TT_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, class_num, task_num):

        super(TNRMTL_TT_net, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num) )
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += TT_Norm(self.cls_weight)
        return regular_term
def TT_Norm(X):
    shapeX = X.size()
    dimX = len(shapeX)
    re = [nuclear_norm(i) for i in [torch.reshape(
        X, [np.prod(shapeX[:j]), np.prod(shapeX[j:])]) for j in range(1, dimX)
                          ]
          ]
    return torch.mean(torch.stack(re))


class TNRMTL_LAF_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, class_num, task_num):

        super(TNRMTL_LAF_net, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)

        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num) )

        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)


    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += LAF_Norm(self.cls_weight)
        return regular_term
def LAF_Norm(X):
    return nuclear_norm(TensorUnfold(X, 0))
class GTTN_weight_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(GTTN_weight_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num))
        self.norm_weight = Parameter(torch.FloatTensor(3))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        #torch.nn.init.kaiming_uniform(self.norm_weight, a=0, mode='fan_in')
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += GTTN_weight_Norm(self.cls_weight,softmax_weight)
        return regular_term

    def get_weight(self):
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        return softmax_weight
def GTTN_weight_Norm(X,norm_weight):
    shapeX = X.size()
    dimX = len(shapeX)
    re = [nuclear_norm(i) for i in [TensorUnfold(X, j) for j in range(dimX)]]
    norm = torch.FloatTensor(re).to(Args.device)


    weight_Norm = torch.mul(norm_weight,norm)
    return torch.sum(weight_Norm)

def nuclear_norm(x):
    _,sigma,_ = torch.svd(x,compute_uv=True)


    return torch.sum(sigma)


class MOLLIE_1_Tucker_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(MOLLIE_1_Tucker_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num))
        self.norm_weight = Parameter(torch.FloatTensor(6))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)

        regular_term += MOLLIE_1_Tucker_Norm(self.cls_weight,self.norm_weight)
        return regular_term
def MOLLIE_1_Tucker_Norm(X,norm_weight):
    K_matrix_list = create_K_list_Tucker(X)
    norm_list = get_norm_from_matrix_list(K_matrix_list)
    m = nn.Softmax(dim=0)
    softmax_weight = m(norm_weight)
    weight_Norm = torch.mul(softmax_weight,torch.FloatTensor(norm_list))
    return torch.sum(weight_Norm)

class MOLLIE_1_TT_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(MOLLIE_1_TT_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num))
        self.norm_weight = Parameter(torch.FloatTensor(4))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += MOLLIE_1_TT_Norm(self.cls_weight,self.norm_weight)
        return regular_term
def MOLLIE_1_TT_Norm(X,norm_weight):
    K_matrix_list = create_K_list_TT(X)
    norm_list = get_norm_from_matrix_list(K_matrix_list)
    m = nn.Softmax(dim=0)
    softmax_weight = m(norm_weight)
    weight_Norm = torch.mul(softmax_weight,torch.FloatTensor(norm_list))
    return torch.sum(weight_Norm)

class MOLLIE_1_LAF_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(MOLLIE_1_LAF_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num))
        self.norm_weight = Parameter(torch.FloatTensor(2))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += MOLLIE_1_LAF_Norm(self.cls_weight,self.norm_weight)
        return regular_term
def MOLLIE_1_LAF_Norm(X,norm_weight):
    K_matrix_list = create_K_list_LAF(X)
    norm_list = get_norm_from_matrix_list(K_matrix_list)
    m = nn.Softmax(dim=0)
    softmax_weight = m(norm_weight)
    weight_Norm = torch.mul(softmax_weight,torch.FloatTensor(norm_list))
    return torch.sum(weight_Norm)

def create_K_list_Tucker(W):
    shape_W =W.size()
    dim_W =len(shape_W)
    A = []
    for j in range(dim_W):
        fold_matrix = TensorUnfold(W, j)
        A.append(RBF_K_matirix(fold_matrix))
        A.append(RBF_K_matirix(torch.transpose(fold_matrix,1,0)))
    return A
def create_K_list_TT(W):
    A = []
    fold_matrix_0 = TensorUnfold(W, 0)
    A.append(RBF_K_matirix(fold_matrix_0))
    A.append(RBF_K_matirix(torch.transpose(fold_matrix_0, 1, 0)))
    fold_matrix_2 = TensorUnfold(W, 2)
    A.append(RBF_K_matirix(fold_matrix_2))
    A.append(RBF_K_matirix(torch.transpose(fold_matrix_2, 1, 0)))
    return A
def create_K_list_LAF(W):
    A = []
    fold_matrix = TensorUnfold(W, 0)
    A.append(RBF_K_matirix(fold_matrix))
    A.append(RBF_K_matirix(torch.transpose(fold_matrix, 1, 0)))
    return A


def RBF_K_matirix(x):
    K_1 = torch.matmul(x,torch.transpose(x,1,0) )
    diag_element = torch.diag(K_1)
    Len = diag_element.size()[0]
    diag_element_matrix= diag_element.expand(Len,Len)
    K = diag_element_matrix + torch.transpose(diag_element_matrix,1,0)- 2*K_1
    K_1 = torch.exp(-torch.div(K,torch.mean(K)))
    return K_1
def get_norm_from_matrix_list(K_list):
    norm_list=[]
    for i in K_list:
        norm_list.append(MOLLIE_nuclear_norm(i))
    return norm_list

def MOLLIE_nuclear_norm(x):
    _, sigma, _ = torch.svd(x, compute_uv=True)
    norm = torch.sum(torch.sqrt(sigma))
    return norm

class MOLLIE_2_Tucker_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(MOLLIE_2_Tucker_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(self.task_num, self.class_num, self.hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(self.task_num, self.class_num))
        # self.weight = Parameter(torch.FloatTensor(6))
        self.norm_weight = Parameter(torch.FloatTensor(6))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        # m = nn.Softmax(dim=0)
       # self.norm_weight = m(self.weight)
        self.net_0 = Parameter(torch.FloatTensor(self.class_num* self.hidden_dim, self.class_num* self.hidden_dim))
        self.net_1 = Parameter(torch.FloatTensor(self.task_num, self.task_num))
        self.net_2 = Parameter(torch.FloatTensor(self.task_num* self.hidden_dim,  self.task_num* self.hidden_dim))
        self.net_3 = Parameter(torch.FloatTensor(self.class_num, self.class_num))
        self.net_4 = Parameter(torch.FloatTensor(self.class_num* self.task_num, self.class_num* self.task_num))
        self.net_5 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        torch.nn.init.normal_(self.net_0, mean=0, std=1)
        torch.nn.init.normal_(self.net_1, mean=0, std=1)
        torch.nn.init.normal_(self.net_2, mean=0, std=1)
        torch.nn.init.normal_(self.net_3, mean=0, std=1)
        torch.nn.init.normal_(self.net_4, mean=0, std=1)
        torch.nn.init.normal_(self.net_5, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += torch.pow(torch.norm(self.cls_weight), 2)
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        regular_term += MOLLIE_2_Tucker_Norm(self.cls_weight,softmax_weight,self.net_0,
                                 self.net_1,self.net_2,self.net_3,self.net_4,self.net_5)
        return regular_term

    def get_weight(self):
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        return softmax_weight
def MOLLIE_2_Tucker_Norm(X,norm_weight,net_0,net_1,net_2,net_3,net_4,net_5):
    norm =[]
    matrix_Tucher_0 = TensorUnfold(X, 0)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_0,net_0))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_0,1,0), net_1))))
    matrix_Tucher_1 = TensorUnfold(X, 1)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_1, net_2))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_1, 1, 0), net_3))))
    matrix_Tucher_2 = TensorUnfold(X, 2)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_2, net_4))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_2, 1, 0), net_5))))

    weight_Norm = torch.mul(norm_weight,torch.FloatTensor(norm).to(Args.device))
    return torch.sum(weight_Norm)

class MOLLIE_2_TT_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(MOLLIE_2_TT_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(self.task_num, self.class_num, self.hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(self.task_num, self.class_num))
        # self.weight = Parameter(torch.FloatTensor(4))
        self.norm_weight = Parameter(torch.FloatTensor(4))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        # m = nn.Softmax(dim=0)
       # self.norm_weight = m(self.weight)
        self.net_0 = Parameter(torch.FloatTensor(self.class_num* self.hidden_dim, self.class_num* self.hidden_dim))
        self.net_1 = Parameter(torch.FloatTensor(self.task_num, self.task_num))
        self.net_2 = Parameter(torch.FloatTensor(self.class_num* self.task_num, self.class_num* self.task_num))
        self.net_3 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        torch.nn.init.normal_(self.net_0, mean=0, std=1)
        torch.nn.init.normal_(self.net_1, mean=0, std=1)
        torch.nn.init.normal_(self.net_2, mean=0, std=1)
        torch.nn.init.normal_(self.net_3, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += torch.pow(torch.norm(self.cls_weight), 2)
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        regular_term += MOLLIE_2_TT_Norm(self.cls_weight,softmax_weight,self.net_0,
                                 self.net_1,self.net_2,self.net_3)
        return regular_term

    def get_weight(self):
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        return softmax_weight
def MOLLIE_2_TT_Norm(X,norm_weight,net_0,net_1,net_2,net_3):
    norm =[]
    matrix_Tucher_0 = TensorUnfold(X, 0)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_0,net_0))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_0,1,0), net_1))))
    matrix_Tucher_1 = TensorUnfold(X, 2)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_1, net_2))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_1, 1, 0), net_3))))

    weight_Norm = torch.mul(norm_weight,torch.FloatTensor(norm).to(Args.device))
    return torch.sum(weight_Norm)

class MOLLIE_2_LAF_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(MOLLIE_2_LAF_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(self.task_num, self.class_num, self.hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(self.task_num, self.class_num))
        # self.weight = Parameter(torch.FloatTensor(2))
        self.norm_weight = Parameter(torch.FloatTensor(2))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        # m = nn.Softmax(dim=0)
       # self.norm_weight = m(self.weight)
        self.net_0 = Parameter(torch.FloatTensor(self.class_num* self.hidden_dim, self.class_num* self.hidden_dim))
        self.net_1 = Parameter(torch.FloatTensor(self.task_num, self.task_num))
        torch.nn.init.normal_(self.net_0, mean=0, std=1)
        torch.nn.init.normal_(self.net_1, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += torch.pow(torch.norm(self.cls_weight), 2)
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        regular_term += MOLLIE_2_LAF_Norm(self.cls_weight,softmax_weight,self.net_0,
                                 self.net_1)
        return regular_term
    def get_weight(self):
        m = nn.Softmax(dim=0)
        softmax_weight = m(self.norm_weight)
        return softmax_weight
def MOLLIE_2_LAF_Norm(X,norm_weight,net_0,net_1):
    norm =[]
    matrix_Tucher_0 = TensorUnfold(X, 0)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_0,net_0))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_0,1,0), net_1))))

    weight_Norm = torch.mul(norm_weight,torch.FloatTensor(norm).to(Args.device))
    return torch.sum(weight_Norm)














class New_GTTN_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(New_GTTN_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(task_num, class_num, hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(task_num, class_num))
        self.norm_weight = Parameter(torch.FloatTensor(3))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        #torch.nn.init.kaiming_uniform(self.norm_weight, a=0, mode='fan_in')
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        norms_need_weight = New_GTTN_Norm(self.cls_weight)
       # print('norms_need_weight')
       # print(len(norms_need_weight))
        # return int, list of int
        return regular_term, norms_need_weight
def New_GTTN_Norm(X):
    shapeX = X.size()
    dimX = len(shapeX)
    norm = [nuclear_norm(i) for i in [TensorUnfold(X, j) for j in range(dimX)]]
    #norm = torch.FloatTensor(re).to(Args.device)
    return norm
def TensorUnfold(A, k):
    A = torch.transpose(A, k,0)
    shapeA = list(A.size())
    return torch.reshape(A, [shapeA[0], np.prod(shapeA[1:])])





class New_MOLLIE_2_Tucker_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(New_MOLLIE_2_Tucker_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(self.task_num, self.class_num, self.hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(self.task_num, self.class_num))
        # self.weight = Parameter(torch.FloatTensor(6))
        # self.norm_weight = Parameter(torch.FloatTensor(6))
        # torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        # m = nn.Softmax(dim=0)
       # self.norm_weight = m(self.weight)
        self.net_0 = Parameter(torch.FloatTensor(self.class_num* self.hidden_dim, self.class_num* self.hidden_dim))
        self.net_1 = Parameter(torch.FloatTensor(self.task_num, self.task_num))
        self.net_2 = Parameter(torch.FloatTensor(self.task_num* self.hidden_dim,  self.task_num* self.hidden_dim))
        self.net_3 = Parameter(torch.FloatTensor(self.class_num, self.class_num))
        self.net_4 = Parameter(torch.FloatTensor(self.class_num* self.task_num, self.class_num* self.task_num))
        self.net_5 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        torch.nn.init.normal_(self.net_0, mean=0, std=1)
        torch.nn.init.normal_(self.net_1, mean=0, std=1)
        torch.nn.init.normal_(self.net_2, mean=0, std=1)
        torch.nn.init.normal_(self.net_3, mean=0, std=1)
        torch.nn.init.normal_(self.net_4, mean=0, std=1)
        torch.nn.init.normal_(self.net_5, mean=0, std=1)
        # try:
        #     output = model(input)
        # except RuntimeError as exception:
        #     if "out of memory" in str(exception):
        #         print("WARNING: out of memory")
        #         if hasattr(torch.cuda, 'empty_cache'):
        #             torch.cuda.empty_cache()
        #     else:
        #         raise exception
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs

    def get_regular_term(self):

        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)

        regular_term += torch.pow(torch.norm(self.cls_weight), 2)

        norms_need_weight = New_MOLLIE_2_Tucker_Norm(self.cls_weight,self.net_0,
                                 self.net_1,self.net_2,self.net_3,self.net_4,self.net_5)
        # return int, list of int
        return regular_term, norms_need_weight

def New_MOLLIE_2_Tucker_Norm(X,net_0,net_1,net_2,net_3,net_4,net_5):
    norm =[]
    matrix_Tucher_0 = TensorUnfold(X, 0)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_0,net_0))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_0,1,0), net_1))))

    matrix_Tucher_1 = TensorUnfold(X, 1)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_1, net_2))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_1, 1, 0), net_3))))

    matrix_Tucher_2 = TensorUnfold(X, 2)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_2, net_4))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_2, 1, 0), net_5))))
    return norm # a list of norms

class New_MOLLIE_2_TT_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(New_MOLLIE_2_TT_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(self.task_num, self.class_num, self.hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(self.task_num, self.class_num))
        # self.weight = Parameter(torch.FloatTensor(4))
        # self.norm_weight = Parameter(torch.FloatTensor(4))
        # torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        # m = nn.Softmax(dim=0)
       # self.norm_weight = m(self.weight)
        self.net_0 = Parameter(torch.FloatTensor(self.class_num* self.hidden_dim, self.class_num* self.hidden_dim))
        self.net_1 = Parameter(torch.FloatTensor(self.task_num, self.task_num))
        self.net_2 = Parameter(torch.FloatTensor(self.class_num* self.task_num, self.class_num* self.task_num))
        self.net_3 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        torch.nn.init.normal_(self.net_0, mean=0, std=1)
        torch.nn.init.normal_(self.net_1, mean=0, std=1)
        torch.nn.init.normal_(self.net_2, mean=0, std=1)
        torch.nn.init.normal_(self.net_3, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += torch.pow(torch.norm(self.cls_weight), 2)
        norms_need_weight = New_MOLLIE_2_TT_Norm(self.cls_weight,self.net_0,
                                 self.net_1,self.net_2,self.net_3)
        # return int, list of int
        return regular_term, norms_need_weight
def New_MOLLIE_2_TT_Norm(X,net_0,net_1,net_2,net_3):
    norm =[]
    matrix_Tucher_0 = TensorUnfold(X, 0)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_0,net_0))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_0,1,0), net_1))))
    matrix_Tucher_1 = TensorUnfold(X, 2)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_1, net_2))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_1, 1, 0), net_3))))

    return norm # a list of norms

class New_MOLLIE_2_LAF_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_num, task_num):
        super(New_MOLLIE_2_LAF_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.task_num = task_num
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_weight = Parameter( torch.FloatTensor(self.task_num, self.class_num, self.hidden_dim) )
        self.cls_bias = Parameter( torch.FloatTensor(self.task_num, self.class_num))
        # self.weight = Parameter(torch.FloatTensor(2))
        self.norm_weight = Parameter(torch.FloatTensor(2))
        torch.nn.init.normal_(self.norm_weight, mean=0, std=1)
        # m = nn.Softmax(dim=0)
       # self.norm_weight = m(self.weight)
        self.net_0 = Parameter(torch.FloatTensor(self.class_num* self.hidden_dim, self.class_num* self.hidden_dim))
        self.net_1 = Parameter(torch.FloatTensor(self.task_num, self.task_num))
        torch.nn.init.normal_(self.net_0, mean=0, std=1)
        torch.nn.init.normal_(self.net_1, mean=0, std=1)
        for i in range(self.task_num):
            init.kaiming_uniform_(self.cls_weight[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cls_weight[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.cls_bias[i], -bound, bound)
    def forward(self, inputs, task_index):
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
        hidden_feature = F.relu( self.hidden_layer(inputs) )
        outputs = F.linear(hidden_feature, self.cls_weight[task_index], self.cls_bias[task_index])
        return outputs
    def get_regular_term(self):
        regular_term = torch.pow(torch.norm(self.hidden_layer.weight), 2)
        regular_term += torch.pow(torch.norm(self.cls_weight), 2)

        norms_need_weight = New_MOLLIE_2_LAF_Norm(self.cls_weight, self.net_0,self.net_1)
        # return int, list of int
        return regular_term, norms_need_weight

def New_MOLLIE_2_LAF_Norm(X,net_0,net_1):
    norm =[]
    matrix_Tucher_0 = TensorUnfold(X, 0)
    norm.append(nuclear_norm(torch.tanh(torch.matmul(matrix_Tucher_0,net_0))))
    norm.append(nuclear_norm(torch.tanh(torch.matmul(torch.transpose(matrix_Tucher_0,1,0), net_1))))
    return norm







class Lambda_net(nn.Module):
    def __init__(self, Wdim):
        """

        :param Wdim: number of regularizers
        """
        assert type(Wdim)==int, "Wdim should be an integer"

        super(Lambda_net, self).__init__()
        self.W =  nn.Parameter(torch.ones(Wdim) * (1.0/Wdim))
        #self.normalized_w = self.W

    def forward(self, is_detach=True):
        #self.normalized_w = torch.softmax(self.W, dim=0)
        self.normalized_w = self.W.pow(2)
        #self.normalized_w = torch.abs(self.W)
        if is_detach:
            return self.normalized_w.detach()
        else:
            return self.normalized_w

class Meta_learner:
    def __init__(self, Wdim, outer_lr,device):
        self.leaning_rate = outer_lr
        self.lambda_net = Lambda_net(Wdim).to(device)
        self.lambda_optim = torch.optim.Adam(self.lambda_net.parameters(),
                                             lr=self.leaning_rate)

    def init_W(self):
        return self.lambda_net(is_detach=True)

    def update_W(self,train_batch,val_batch,
                 inner_model,inner_optimizer,
                 num_class, num_task, Args, batch_size):
        """
        regularizer_list: list of regularizer class,
            which contains a get_reguxxx function
        """
        # train_batch is an Iterator
        # val_batch also
        #sampled_data, sampled_label, sampled_task_ind, _ = Iterator.get_next_batch()
        #print('[update_W]: num_class, num_task, batch_size ', num_class, num_task, batch_size )
        class_criterion = nn.CrossEntropyLoss()

        w_len = self.lambda_net.W.shape[0]
        usual_norm, norms_need_weight = inner_model.get_regular_term()
        regularizer_len = len(norms_need_weight)
        # print(regularizer_len)
        # print(w_len)
        assert w_len == regularizer_len, "The length of meta parameter should" +\
            "equals to number of regularizers."
        #print('23924983257', next(inner_model.parameters()).is_cuda ) # True

        import higher
        with higher.innerloop_ctx(inner_model, inner_optimizer) as (fmodel, diffopt):
            sampled_data, sampled_label, sampled_task_ind, _ = train_batch.get_next_batch()
            # sampled_data [xxx, feature_size]
            # sampeled_label [n*batchsize, num_class]
            # sampled_task_ind []
            usual_norm, norms_need_weight = fmodel.get_regular_term()

            empirical_loss = None
            for i in range(num_task):
                x = torch.tensor(sampled_data[i * num_class * batch_size:
                                (i + 1) * num_class * batch_size]).to(Args.device)
                y_hat = fmodel(x, i)
                #outputs.append(y_hat)

                y_this_task_one_hot = torch.tensor(sampled_label[i * num_class * batch_size:
                                            (i + 1) * num_class * batch_size]).to(Args.device)


                y_this_task_int = y_this_task_one_hot.argmax(dim=1)

                #print('y_hat.shape, y_this_task_int.shape: ', y_hat.shape, y_this_task_int.shape)
                loss_this_task = class_criterion(y_hat, y_this_task_int)

                if empirical_loss is None:
                    empirical_loss = loss_this_task
                else:
                    empirical_loss += loss_this_task
            del sampled_data, sampled_label, sampled_task_ind


            # add regulurizer term, introducing the w
            w = self.lambda_net(is_detach=False)
            reg_term = 0.0
            for reg_idx in range(w_len):
                reg_loss_this = w[reg_idx] * norms_need_weight[reg_idx]
                reg_term += reg_loss_this

            total_l = empirical_loss + reg_term
            diffopt.step(total_l)

            # here, we got the theta t+1 !!!
            val_x, val_y_onehot, sampled_task_ind, _ = val_batch.get_next_batch()
            # for validation set


            self.lambda_optim.zero_grad()
            val_empirical_loss = None
            for i in range(num_task):
                val_y_hat = fmodel(torch.tensor(val_x[i * num_class * batch_size:
                                            (i + 1) * num_class * batch_size]).to(Args.device), i)
                y_this_task_one_hot_val = torch.tensor(val_y_onehot[i * num_class * batch_size:
                                            (i + 1) * num_class * batch_size]).to(Args.device)
                y_this_task_int = y_this_task_one_hot.argmax(dim=1)
                val_loss_this_task = class_criterion(val_y_hat, y_this_task_int)

                if val_empirical_loss is None:
                    val_empirical_loss = val_loss_this_task
                else:
                    val_empirical_loss += val_loss_this_task
            # update meta parameter using the validation empirical loss
            val_empirical_loss.backward()
            self.lambda_optim.step()

            return self.lambda_net(is_detach=True)


