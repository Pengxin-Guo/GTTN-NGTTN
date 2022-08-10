import numpy as np
import random
import re
#from model import DMTL_net_
import args as Args
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR


class MTDataset:
    def __init__(self, data, label, task_interval, num_class, batch_size):
        self.data = data
        self.data_dim = data.shape[1]
        self.label = np.reshape(label, [1, -1])
        self.task_interval = np.reshape(task_interval, [1, -1])
        self.num_task = task_interval.size-1
        self.num_class = num_class
        self.batch_size = batch_size
        self.__build_index__()

    def __build_index__(self):
        index_list = []
        for i in range(self.num_task):
            start = self.task_interval[0, i]
            end = self.task_interval[0, i+1]
            for j in range(self.num_class):
                index_list.append(np.arange(start,end)[np.where(self.label[0, start:end]==j)[0]])
        self.index_list = index_list
        self.counter = np.zeros([1,self.num_task*self.num_class], dtype=np.int32)

    def get_next_batch(self):

        sampled_data = np.zeros([self.batch_size*self.num_class*self.num_task, self.data_dim], dtype=np.float32)
        sampled_label = np.zeros([self.batch_size*self.num_class*self.num_task, self.num_class], dtype=np.int32)
        sampled_task_ind = np.zeros([1, self.batch_size*self.num_class*self.num_task], dtype=np.int32)
        sampled_label_ind = np.zeros([1, self.batch_size*self.num_class*self.num_task], dtype=np.int32)
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i*self.num_class+j
                task_class_index = self.index_list[cur_ind]
                #print('task_class_index = ', task_class_index)
                sampled_ind = range(cur_ind*self.batch_size,(cur_ind+1)*self.batch_size)
                sampled_task_ind[0, sampled_ind] = i
                sampled_label_ind[0, sampled_ind] = j
                sampled_label[sampled_ind, j] = 1
                if task_class_index.size<self.batch_size:
                    sampled_data[sampled_ind, :] = self.data[np.concatenate((task_class_index, task_class_index[np.random.randint(0, high=task_class_index.size,
                                                                                                                 size=self.batch_size-task_class_index.size)])),:]
                elif self.counter[0, cur_ind]+self.batch_size < task_class_index.size:
                    sampled_data[sampled_ind,:] = self.data[task_class_index[self.counter[0, cur_ind]:self.counter[0, cur_ind]+self.batch_size],:]
                    self.counter[0, cur_ind] = self.counter[0, cur_ind] + self.batch_size
                else:
                    sampled_data[sampled_ind, :] = self.data[task_class_index[-self.batch_size:],:]
                    self.counter[0, cur_ind] = 0
                    np.random.shuffle(self.index_list[cur_ind])
        return sampled_data, sampled_label, sampled_task_ind, sampled_label_ind
class MTDataset_Split:
    def __init__(self, data, label, task_interval, num_class):
        self.data = data
        self.data_dim = data.shape[1]
        self.label = np.reshape(label, [1, -1])
        self.task_interval = np.reshape(task_interval, [1, -1])
        self.num_task = task_interval.size-1
        self.num_class = num_class
        self.__build_index__()

    def __build_index__(self):
        index_list = []
        self.num_class_ins = np.zeros([self.num_task, self.num_class])
        for i in range(self.num_task):
            start = self.task_interval[0, i]
            end = self.task_interval[0, i+1]
            for j in range(self.num_class):
                index_array = np.where(self.label[0, start:end]==j)[0]
                self.num_class_ins[i,j] = index_array.size
                index_list.append(np.arange(start,end)[index_array])
        self.index_list = index_list

    def split(self, train_size):

        if train_size < 1:
            train_num = np.ceil(self.num_class_ins * train_size).astype(np.int32)
        else:
            train_num = np.ones([self.num_task, self.num_class], dtype=np.int32) * train_size
            train_num = np.maximum(1, np.minimum(train_num, self.num_class_ins - 10))
        traindata = np.zeros([0, self.data_dim], dtype=np.float32)
        testdata = np.zeros([0, self.data_dim], dtype=np.float32)
        trainlabel = np.zeros([1, 0], dtype=np.int32)
        testlabel = np.zeros([1, 0], dtype=np.int32)
        train_task_interval = np.zeros([1, self.num_task+1], dtype=np.int32)
        test_task_interval = np.zeros([1, self.num_task+1], dtype=np.int32)
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i * self.num_class + j
                task_class_index = self.index_list[cur_ind]
                np.random.shuffle(task_class_index)
                train_index = task_class_index[0:train_num[i,j]]
                test_index = task_class_index[train_num[i,j]:]
                traindata = np.concatenate((traindata, self.data[train_index,:]), axis=0)
                trainlabel = np.concatenate((trainlabel, np.ones([1, train_index.size], dtype=np.int32)*j), axis=1)
                testdata = np.concatenate((testdata, self.data[test_index,:]), axis=0)
                testlabel = np.concatenate((testlabel, np.ones([1, test_index.size], dtype=np.int32)*j), axis=1)
            train_task_interval[0, i+1] = trainlabel.size
            test_task_interval[0, i+1] = testlabel.size
        return traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval

def read_data_from_file(filename):
    file = open(filename, 'r')
    contents = file.readlines()
    file.close()
    num_task = int(contents[0])
    num_class = int(contents[1])
    temp_ind = re.split(',', contents[2])
    temp_ind = [int(elem) for elem in temp_ind]
    task_interval = np.reshape(np.array(temp_ind), [1, -1])
    temp_data = []
    for pos in range(3, len(contents)-1):
        temp_sub_data = re.split(',', contents[pos])
        temp_sub_data = [float(elem) for elem in temp_sub_data]
        temp_data.append(temp_sub_data)
    data = np.array(temp_data)
    temp_label = re.split(',', contents[-1])
    temp_label = [int(elem) for elem in temp_label]
    label = np.reshape(np.array(temp_label), [1, -1])
    return data, label, task_interval, num_task, num_class

def generate_label_task_ind(label, task_interval, num_class):
    num_task = task_interval.size - 1
    num_ins = label.size
    label_matrix = np.zeros((num_ins, num_class), dtype=np.int32)
    label_matrix[range(num_ins), label] = 1
    task_ind = np.zeros((1, num_ins), dtype=np.int32)
    for i in range(num_task):
        task_ind[0, task_interval[0, i] : task_interval[0, i + 1]] = i
    return label_matrix, task_ind

def compute_errors(output, task_ind, label, num_task):
    num_total_ins = output.shape[0]
    num_ins = np.zeros([1, num_task])
    errors = np.zeros([1, num_task + 1])
    for i in range(num_total_ins):
        probit = output[i, :]
        num_ins[0, task_ind[0, i]] += 1
        if np.argmax(probit) != label[0, i]:
            errors[0, task_ind[0, i]] += 1
    for i in range(num_task):
        errors[0, i] = errors[0, i] / num_ins[0, i]
    errors[0, num_task] = np.mean(errors[0, 0:num_task])
    return errors



def train_teacher(model_instance, traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval,
                  validationdata, validationlabel, validation_task_interval,
                num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,outer_learning_rate,filename,
                  freq_update_w,num_norm):

    model_instance.train()
    max_iter_epoch = np.ceil(traindata.shape[0] / (batch_size * num_task * num_class)).astype(np.int32)

    Iterator_train = MTDataset(traindata, trainlabel, train_task_interval,
                               num_class, batch_size)
    Iterator_val = MTDataset(validationdata, validationlabel, validation_task_interval,
                               num_class, batch_size)

    running_loss = 0.0
    running_cls_loss = 0.0
    Test_error = np.ones([1, num_task + 1])

    # for meta learning
    from model import Meta_learner
    meta_learner = Meta_learner(num_norm,outer_learning_rate, Args.device)
    #freq_update_w = 1
    freq_print_w = 15
    #init w
    reg_W = meta_learner.init_W().to(Args.device)
    norm_weight =reg_W.cpu()
    print(norm_weight)
    num_iter_max =0

    for iter in range(max_iter_epoch * max_epoch):

        sampled_data_train, sampled_label_train, sampled_task_ind_train, _ = Iterator_train.get_next_batch()
        num_iter = iter // max_iter_epoch                   #epoch = num_iter
        #lr = learning_rate / (1 + num_iter)
        optimizer = optim.Adam(model_instance.parameters(), lr= learning_rate / (1 + num_iter))


        class_criterion = nn.CrossEntropyLoss()
        outputs = []
        for i in range(num_task):
            output = model_instance(torch.tensor(sampled_data_train[i * num_class * batch_size:
                                            (i + 1) * num_class * batch_size]).to(Args.device), i)
            outputs.append(output)
        outputs = torch.cat(outputs, 0)
        sampled_label_train = torch.tensor([np.where(sampled_label_train[_] == 1)[0][0] for _ in range(sampled_label_train.shape[0])]).to(Args.device)
        cls_loss = class_criterion(outputs, sampled_label_train) * num_task
        usual_norm, norms_need_weight = model_instance.get_regular_term()

        norms_number = len(norms_need_weight)
        weighted_norms = sum([reg_W[i]* norms_need_weight[i] \
                              for i in range(norms_number)])
        obj = cls_loss + reg_para * (usual_norm + weighted_norms)

        optimizer.zero_grad()
        obj.backward()
        optimizer.step()


        running_cls_loss = running_cls_loss + cls_loss.item()
        running_loss = running_loss + obj.item()
        tmp = iter + 1

        if tmp  %  max_iter_epoch == 0 :



            test_error = test(model_instance, testdata, testlabel, test_task_interval)
            test_accuracy = 1 - test_error
            print(str(model_instance)[0:10]+' epoch %d, cls_loss %g, objective value %g test_accuracy = %g' %
                  (num_iter, running_cls_loss / max_iter_epoch, running_loss / max_iter_epoch,test_accuracy[0][num_task]))
            if (test_error[0][num_task] < Test_error[0][num_task]):
                Test_error = test_error
                num_iter_max = num_iter
                norm_weight = reg_W.cpu()

            running_cls_loss = 0.0
            running_loss = 0.0

        if iter % freq_update_w == 0:
            reg_W = meta_learner.update_W(
                train_batch = Iterator_train,
                val_batch =Iterator_val,
                inner_model = model_instance,
                inner_optimizer =optimizer,
                num_class = num_class ,
                num_task =num_task,
                Args = Args,
                batch_size = batch_size )
            #print('update W!')
        if iter % freq_print_w == 0:

            print(reg_W)
            print(norm_weight)
            print(num_iter_max)
            print('max_acc: ')
            print(1-Test_error[0][num_task])



    return Test_error


def train_teacher_baseline(model_instance, traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval,
                        num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,filename):

    model_instance.train()
    max_iter_epoch = np.ceil(traindata.shape[0] / (batch_size * num_task * num_class)).astype(np.int32)
    Iterator = MTDataset(traindata, trainlabel, train_task_interval, num_class, batch_size)

    running_loss = 0.0
    running_cls_loss = 0.0
    Test_error = np.ones([1, num_task + 1])
    for iter in range(max_iter_epoch * max_epoch):
        sampled_data, sampled_label, sampled_task_ind, _ = Iterator.get_next_batch()
        num_iter = iter // max_iter_epoch                   #epoch = num_iter
        #lr = learning_rate / (1 + num_iter)

        optimizer = optim.Adam(model_instance.parameters(), lr= learning_rate / (1 + num_iter))

        scheduler = CosineAnnealingLR(optimizer, T_max=int(10e5), eta_min=5e-6)

        class_criterion = nn.CrossEntropyLoss()
        outputs = []
        for i in range(num_task):
            output = model_instance(torch.tensor(sampled_data[i * num_class * batch_size:
                                            (i + 1) * num_class * batch_size]).to(Args.device), i)
            outputs.append(output)
        outputs = torch.cat(outputs, 0)
        sampled_label = torch.tensor([np.where(sampled_label[_] == 1)[0][0] for _ in range(sampled_label.shape[0])]).to(Args.device)
        cls_loss = class_criterion(outputs, sampled_label) * num_task
        obj = cls_loss + reg_para * model_instance.get_regular_term()

        optimizer.zero_grad()
        obj.backward()
        optimizer.step()
        scheduler.step()
        running_cls_loss = running_cls_loss + cls_loss.item()
        running_loss = running_loss + obj.item()
        tmp = iter + 1
        if tmp  %  max_iter_epoch == 0 :



            test_error = test(model_instance, testdata, testlabel, test_task_interval)
            test_accuracy = 1 - test_error
            print(str(model_instance)[0:10]+' epoch %d, cls_loss %g, objective value %g test_accuracy = %g' %
                  (num_iter, running_cls_loss / max_iter_epoch, running_loss / max_iter_epoch,test_accuracy[0][num_task]))
            if (test_error[0][num_task] < Test_error[0][num_task]):
                Test_error = test_error

            running_cls_loss = 0.0
            running_loss = 0.0



    return Test_error

def train_teacher_baseline_1(model_instance, traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval,
                        num_task, num_class, batch_size, reg_para, max_epoch, learning_rate,filename):

    model_instance.train()
    max_iter_epoch = np.ceil(traindata.shape[0] / (batch_size * num_task * num_class)).astype(np.int32)
    Iterator = MTDataset(traindata, trainlabel, train_task_interval, num_class, batch_size)

    running_loss = 0.0
    running_cls_loss = 0.0
    Test_error = np.ones([1, num_task + 1])
    norm_weight = model_instance.get_weight()
    for iter in range(max_iter_epoch * max_epoch):

        sampled_data, sampled_label, sampled_task_ind, _ = Iterator.get_next_batch()
        num_iter = iter // max_iter_epoch                   #epoch = num_iter
        #lr = learning_rate / (1 + num_iter)

        optimizer = optim.Adam(model_instance.parameters(), lr= learning_rate / (1 + num_iter))

        scheduler = CosineAnnealingLR(optimizer, T_max=int(10e5), eta_min=5e-6)

        class_criterion = nn.CrossEntropyLoss()
        outputs = []
        for i in range(num_task):
            output = model_instance(torch.tensor(sampled_data[i * num_class * batch_size:
                                            (i + 1) * num_class * batch_size]).to(Args.device), i)
            outputs.append(output)
        outputs = torch.cat(outputs, 0)
        sampled_label = torch.tensor([np.where(sampled_label[_] == 1)[0][0] for _ in range(sampled_label.shape[0])]).to(Args.device)
        cls_loss = class_criterion(outputs, sampled_label) * num_task
        obj = cls_loss + reg_para * model_instance.get_regular_term()

        optimizer.zero_grad()
        obj.backward()
        optimizer.step()
        scheduler.step()
        running_cls_loss = running_cls_loss + cls_loss.item()
        running_loss = running_loss + obj.item()
        tmp = iter + 1
        if tmp  %  max_iter_epoch == 0 :
            print(norm_weight)



            test_error = test(model_instance, testdata, testlabel, test_task_interval)
            test_accuracy = 1 - test_error
            print(str(model_instance)[0:10]+' epoch %d, cls_loss %g, objective value %g test_accuracy = %g' %
                  (num_iter, running_cls_loss / max_iter_epoch, running_loss / max_iter_epoch,test_accuracy[0][num_task]))
            if (test_error[0][num_task] < Test_error[0][num_task]):
                Test_error = test_error

            running_cls_loss = 0.0
            running_loss = 0.0


    return Test_error


def test(model_instance, testdata, testlabel, test_task_interval):

    model_instance.eval()
    _, test_task_ind = generate_label_task_ind(testlabel, test_task_interval, model_instance.class_num)
    softmax_outputs = []
    for i in range(testdata.shape[0]):
        softmax = nn.Softmax(dim=1)
        softmax_output = softmax( model_instance( torch.tensor(testdata[i], dtype=torch.float).to(Args.device),
                               test_task_ind[0, i] ) )
        softmax_outputs.append(softmax_output)
    all_probs = torch.cat(softmax_outputs, 0)
    test_error = compute_errors(all_probs.cpu().detach().numpy(), test_task_ind, testlabel, model_instance.task_num)
    
    return test_error