
import torch
import numpy as np
import function as my_F
import args as Args
from model import DMTL_net_,TNRMTL_Tucker_net,TNRMTL_TT_net,TNRMTL_LAF_net,\
    GTTN_weight_net,New_GTTN_net,MOLLIE_1_Tucker_net,MOLLIE_1_TT_net,MOLLIE_1_LAF_net,\
    MOLLIE_2_Tucker_net,MOLLIE_2_TT_net,MOLLIE_2_LAF_net,New_MOLLIE_2_Tucker_net,New_MOLLIE_2_TT_net,New_MOLLIE_2_LAF_net
#from torchsummary import summary
import random
import time

def main_process(filename, train_size, hidden_dim, batch_size, reg_para, max_epoch, learning_rate,outer_learning_rate,
                  freq_update_w):

    print(filename)
    print(Args.device)
    print(train_size)
    print(reg_para)
    print(hidden_dim)
    method_list =[]
    data, label, task_interval, num_task, num_class = my_F.read_data_from_file(filename)
    input_dim = data.shape[1]
    data_split = my_F.MTDataset_Split(data, label, task_interval, num_class)
    traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval = data_split.split(train_size)
    del data_split
    sub_errors = np.zeros([0, num_task + 1])
    print(time.asctime( time.localtime(time.time()) ))


    #
    # DMTL = DMTL_net_(input_dim, hidden_dim, num_class, num_task).to(Args.device)
    # sub_error = my_F.train_teacher_baseline(DMTL, traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval,
    #            num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,filename)
    # sub_errors = np.concatenate((sub_errors, sub_error), axis=0)
    #
    # method_list.append('DMTL')
    # print(time.asctime(time.localtime(time.time())))
    #
    # # del DMTL
    #
    # TNRMTL_Tucker = TNRMTL_Tucker_net(input_dim, hidden_dim, num_class, num_task).to(Args.device)
    # sub_error =my_F.train_teacher_baseline(TNRMTL_Tucker, traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval,
    #                    num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,filename)
    # sub_errors = np.concatenate((sub_errors, sub_error), axis=0)
    # method_list.append('TNRMTL_Tucker')
    # print(time.asctime(time.localtime(time.time())))

    # del TNRMTL_Tucker
    #
    # TNRMTL_TT = TNRMTL_TT_net(input_dim, hidden_dim, num_class, num_task).to(Args.device)
    # sub_error = my_F.train_teacher_baseline(TNRMTL_TT, traindata, trainlabel, train_task_interval, testdata, testlabel,
    #                    test_task_interval,
    #                    num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,filename)
    # sub_errors = np.concatenate((sub_errors, sub_error), axis=0)
    # method_list.append('TNRMTL_TT')
    # print(time.asctime(time.localtime(time.time())))

    # del TNRMTL_TT
    # TNRMTL_LAF = TNRMTL_LAF_net(input_dim, hidden_dim, num_class, num_task).to(Args.device)
    # sub_error = my_F.train_teacher_baseline(TNRMTL_LAF, traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval,
    #                    num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,filename)
    # sub_errors = np.concatenate((sub_errors, sub_error), axis=0)
    # print(time.asctime(time.localtime(time.time())))
    # method_list.append('TNRMTL_LAF')

    # del TNRMTL_LAF
    #
    GTTN_weight = GTTN_weight_net(input_dim, hidden_dim, num_class, num_task).to(Args.device)
    sub_error = my_F.train_teacher_baseline_1(GTTN_weight, traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval,
                       num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,filename)
    sub_errors = np.concatenate((sub_errors, sub_error), axis=0)
    method_list.append('GTTN_weight')



    # data_split_validation = my_F.MTDataset_Split(traindata, trainlabel, train_task_interval, num_class)
    # traindata, trainlabel, train_task_interval, validationdata, validationlabel, validation_task_interval = data_split_validation.split(0.7)
    #
    #
    #
    # New_GTTN = New_GTTN_net(input_dim, hidden_dim, num_class, num_task).to(Args.device)
    # sub_error = my_F.train_teacher(New_GTTN, traindata, trainlabel, train_task_interval, testdata, testlabel,
    #                                test_task_interval, validationdata, validationlabel, validation_task_interval,
    #                                num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,outer_learning_rate, filename,
    #                                 freq_update_w,num_norm=3)
    # sub_errors = np.concatenate((sub_errors, sub_error), axis=0)
    # method_list.append('New_GTTN ')








    Accuracy = np.ones_like(sub_errors)- sub_errors

    print(method_list)
    print(Accuracy)
    ACC_list = Accuracy[:,num_task].tolist()
    ACC_list_4=[]
    for iii in range(len(ACC_list)):
        ACC_list_4.append(round(ACC_list[iii],4))
    print(filename)
    print(ACC_list_4)

    return


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
for Round in range(1):

    for ii in [0.0]:
        for freq_update_w in [5]:
            main_process(Args.datafile, Args.train_size + ii, Args.hidden_dim, Args.batch_size, Args.reg_para,
                         Args.max_epoch, Args.learning_rate,
             Args.outer_learning_rate,  freq_update_w)

