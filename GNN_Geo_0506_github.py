'''
shichang ding
pyg based gnn-geo

sorry do not have enough time to make this code more friendly or more formal, too busy recently
and most code is finished two years ago, i forgot many details..
...at that time i never thought this work need almost 1.5 year to be accepted....
'''
import torch
import os
import numpy as np
import random as rd
def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
setup_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--node_embedding_size', type=int, default=256)
parser.add_argument('--node_hidden_size', type=int, default=128)
parser.add_argument('--edge_hidden_size', type=int, default=16)
parser.add_argument('--fc_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0100, help='Learning rate.')
parser.add_argument('--aggregator_type', type=str, default='add',
                    help='max,add,mean')  #
parser.add_argument('--print_flag', type=int, default=1)
parser.add_argument('--num_step_message_passing', type=int, default=2)  #
parser.add_argument('--regs', type=float, default=0.0005)  # l2
parser.add_argument('--gnn_dropout', type=float, default=0)  # dp not used in GNN-Geo
parser.add_argument('--edge_dropout', type=float, default=0)  #
parser.add_argument('--nn_dropout1', type=float, default=0)  #
args = parser.parse_args()

use_cuda = args.gpuid >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpuid)
else:
    device = "cpu"# i have not tested cpu version
print(use_cuda)

import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import NNConv
# from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from sklearn import preprocessing
from math import radians, cos, sin, asin, sqrt
import itertools
import warnings
warnings.filterwarnings("ignore", category=Warning)

aggregator_type = args.aggregator_type
node_embedding_size = args.node_embedding_size  # [64,32,16,4,1]
node_hidden_size = args.node_hidden_size
edge_hidden_size = args.edge_hidden_size
num_step_message_passing = args.num_step_message_passing
fc_size = args.fc_size
regs = args.regs
cur_best_pre_0 = np.inf
stopping_step = 0
gnn_dropout = args.gnn_dropout
edge_dropout = args.edge_dropout
nn_dropout1 = args.nn_dropout1

target_scaler1 = preprocessing.MinMaxScaler()
target_scaler2 = preprocessing.MinMaxScaler()

def get_label_for_GNN_Geo(filepath,node_num):#借用gnn-geo的train test val划分和label压缩代码

        fr3 = open(filepath, 'r', encoding='UTF-8')
        target_lat_label_dict = {}
        target_lon_label_dict = {}

        label_lat_array = []
        label_lon_array = []

        label_list = [[-99,-99]] * node_num
        label_list_scaled = [[-99, -99]] * node_num

        for line in fr3.readlines():
            str_list = line.strip('\r\n').split(sep=',')
            node_id = int(str_list[1])

            lat_label = float(str_list[4])
            lon_label = float(str_list[3])#
            label_list[node_id]=[lon_label,lat_label]#
            label_lat_array.append(lat_label)
            label_lon_array.append(lon_label)
            target_lat_label_dict[node_id] = lat_label
            target_lon_label_dict[node_id] = lon_label
        #
        train_node_id_list = []  # 7:2:1
        lat_train_node_label_list = []
        lon_train_node_label_list = []

        val_node_id_list = []
        lat_val_node_label_list = []
        lon_val_node_label_list = []

        test_node_id_list = []
        lat_test_node_label_list = []
        lon_test_node_label_list = []

        for key, item in target_lat_label_dict.items():
            node_id = int(key)
            lat_label = target_lat_label_dict[node_id]
            lon_label = target_lon_label_dict[node_id]

            rd_number = rd.random()
            if (rd_number < 0.1):
                train_node_id_list.append(node_id)
                lat_train_node_label_list.append(lat_label)
                train_ratio = args.train_ratio
                if (rd_number < train_ratio):
                    train_node_id_list.append(node_id)
                    lat_train_node_label_list.append(lat_label)
                    lon_train_node_label_list.append(lon_label)

                elif (rd_number < (train_ratio + 0.2)) & (rd_number >= train_ratio):
                    val_node_id_list.append(node_id)
                    lat_val_node_label_list.append(lat_label)
                    lon_val_node_label_list.append(lon_label)

                else:
                    test_node_id_list.append(node_id)
                    lat_test_node_label_list.append(lat_label)
                    lon_test_node_label_list.append(lon_label)

        new_train_label_lat_array = target_scaler1.fit_transform(np.array(lat_train_node_label_list).reshape(-1,1))  #
        new_train_label_lon_array = target_scaler2.fit_transform(np.array(lon_train_node_label_list).reshape(-1, 1))

        new_train_label_lat_array = new_train_label_lat_array.astype(np.float32)  #
        new_train_label_lon_array = new_train_label_lon_array.astype(np.float32)  #

        #

        new_val_label_lat_array = target_scaler1.transform(
            np.array(lat_val_node_label_list).reshape(-1, 1))  #
        new_val_label_lon_array = target_scaler2.transform(np.array(lon_val_node_label_list).reshape(-1, 1))

        new_val_label_lat_array = new_val_label_lat_array .astype(
            np.float32)  #
        new_val_label_lon_array = new_val_label_lon_array.astype(
            np.float32)  #

        new_test_label_lat_array = target_scaler1.transform(
            np.array(lat_test_node_label_list).reshape(-1, 1))  #
        new_test_label_lon_array = target_scaler2.transform(np.array(lon_test_node_label_list).reshape(-1, 1))

        new_test_label_lat_array = new_test_label_lat_array.astype(
            np.float32)  #
        new_test_label_lon_array = new_test_label_lon_array.astype(
            np.float32)  #

        #
        for i,node_id in enumerate(train_node_id_list):
            label_list_scaled[node_id] =[new_train_label_lon_array[i][0],new_train_label_lat_array[i][0]]
        for i,node_id in enumerate(val_node_id_list):
            label_list_scaled[node_id] =[lon_val_node_label_list[i],lat_val_node_label_list[i]]
        for i,node_id in enumerate(test_node_id_list):
            label_list_scaled[node_id] = [lon_test_node_label_list[i],lat_test_node_label_list[i]]

        label_list_scaled = torch.tensor(label_list_scaled)

        return [label_list_scaled,train_node_id_list,val_node_id_list,test_node_id_list]

def mydataset(filepath):
        fr1 = open(filepath+'/data_0419/bj_ip_feature.txt', 'r')
        node_feature_list = []
        for line in fr1.readlines():
            # ip1,0,0.0,0.6901408450704225,0.4645669291338583,0.2784313725490196,0.4031620553359684
            temp_list = line.strip('\n').split(',')[2:]
            temp_list = list(np.float_(temp_list))
            node_feature_list.append(temp_list)
        num_features = len(temp_list)#节点特征维度
        x = torch.tensor(node_feature_list)
        # x2 = torch.tensor([[-1,0], [0,2], [1,1]], dtype=torch.float)
        node_num = len(x)

        #  bj_edge_feature.txt
        filepath = os.path.split(os.path.realpath(__file__))[0]
        fr2 = open(filepath + '/data_0419/bj_edge_feature.txt', 'r')
        edge_list = []
        edge_feature_list = []
        for line in fr2.readlines():
            # 0,2,0.49998416488906056,0.0,0.6901408450704225,0.4645669291338583,0.2784313725490196,0.4031620553359684,0.16431924882629106,0.4763779527559055,0.20392156862745098,0.0
            temp_list = line.strip('\n').split(',')
            temp_edge1 = [int(temp_list[0]), int(temp_list[1])]
            temp_edge2 = [int(temp_list[1]), int(temp_list[0])]  # 无向图需要重复一遍
            edge_list.append(temp_edge1)
            edge_list.append(temp_edge2)
            temp_feature_list = temp_list[2:]
            temp_feature_list = list(np.float_(temp_feature_list))
            #
            edge_feature_list.append(temp_feature_list)
            edge_feature_list.append(temp_feature_list)
        edge_index = torch.tensor(edge_list)
        edge_attr = torch.tensor(edge_feature_list)
        # x2 = torch.tensor([[-1,0], [0,2], [1,1]], dtype=torch.float)
        edge_num = len(edge_index) / 2

        #
        y, train_node_id_list, val_node_id_list, test_node_id_list = get_label_for_GNN_Geo(filepath=filepath+'/data_0419/bj_dstip_id_allinfo.txt', node_num=node_num)

        train_mask = torch.tensor(train_node_id_list)
        val_mask = torch.tensor(val_node_id_list)
        test_mask = torch.tensor(test_node_id_list)

        graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, train_mask=train_mask, val_mask=val_mask,
                     test_mask=test_mask,edge_attr=edge_attr,num_features=num_features)
        print(graph)
        data_list = [graph]
        return [data_list,train_node_id_list,val_node_id_list,test_node_id_list]


class MPNN(torch.nn.Module):
    def __init__(self, aggregator_type,node_in_feats, node_hidden_dim, edge_input_dim, edge_hidden_dim,num_step_message_passing,gconv_dp,edge_dp,nn_dp1):
        super(MPNN, self).__init__()
        self.lin0 = torch.nn.Linear(node_in_feats, node_hidden_dim)#
        self.num_step_message_passing = num_step_message_passing
        # nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))#
        edge_network = Sequential(
            #

            Linear(edge_input_dim, edge_hidden_dim),
            nn.Dropout(p=edge_dp),
            ReLU(),
            Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
            )  # 1-4-32x32

        self.conv = NNConv(node_hidden_dim, node_hidden_dim, edge_network, aggr=aggregator_type)#

        self.y_linear = nn.Linear(node_hidden_dim, 2)  # 4-4
        self.bn = nn.BatchNorm1d(node_hidden_dim)
        self.gnn_dropout = nn.Dropout(p=gconv_dp)#dropout
        self.nn_dropout = nn.Dropout(p=nn_dp1)

    def forward(self, data):
        out = torch.relu(self.lin0(data.x))  # (B1, H1)
        data.edge_attr = torch.tensor(data.edge_attr, dtype=torch.float32)#
        for i in range(self.num_step_message_passing):
            out = torch.relu(self.conv(out, data.edge_index ,data.edge_attr ))
            # (out, data.edge_index, data.edge_attr)
            out = self.gnn_dropout(out)# (B1, H1)

        y_bn = self.bn(out)
        # y_bn = y_bn)
        y_sigmoid = torch.sigmoid(self.nn_dropout(self.y_linear(y_bn)))

        return y_sigmoid



def geodistance(lng1, lat1, lng2, lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=3):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']
    # acc
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step,
                                                                    best_value))  # best_value
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2

if __name__ == '__main__':

    # 数据集对象操作
    filepath = os.path.split(os.path.realpath(__file__))[0]
    dataset,train_node_id_list,val_node_id_list,test_node_id_list = mydataset(filepath)
    data=dataset[0]

    node_id_embedding_layer = nn.Embedding(len(data.x),
                                           node_embedding_size)  # 23216 x 16
    node_attr = torch.tensor(data.x, dtype=torch.float)  #
    id_tensor = torch.LongTensor(list(range(len(data.x))))
    node_id_embedding = node_id_embedding_layer(id_tensor)
    x = torch.cat([node_id_embedding, node_attr], dim=1)

    model = MPNN(aggregator_type, node_embedding_size + 5, node_hidden_size, 10, edge_hidden_size,
               num_step_message_passing,gnn_dropout, edge_dropout,
               nn_dropout1)#
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), node_id_embedding.parameters()), lr=args.lr,weight_decay=regs)

    #----save file-------------------------------
    from time import time
    stamp = int(time())
    result_path1 = filepath + '/output_1110_pyg_gnn_geo_bj/' + str(stamp) + '_loss.txt'  # 详细loss 演化过程
    if not os.path.exists("./output_1110_pyg_gnn_geo_bj/"):
         os.makedirs("./output_1110_pyg_gnn_geo_bj/")
    fw1 = open(result_path1, 'w')
    perf_str = 'aggregator_type=%s,regs=%s,node_embedding_size=%s, node_hidden_size=%s,edge_hidden_size=%s,fc_size=%s,num_step_message_passing=%s, lr=%.4f\n' \
                % (aggregator_type, args.regs,
                   args.node_embedding_size, args.node_hidden_size, args.edge_hidden_size, args.fc_size,
                   args.num_step_message_passing, args.lr)

    fw1.write(perf_str)
    if args.print_flag > 0:
         print(perf_str)
    fw1.flush()
    #----save file-------------------------------
    loss_fn = nn.MSELoss()
    if use_cuda:
        loss_fn=loss_fn.cuda()
    for epoch in range(1, args.epochs + 1):

        model.train()
        out = model(data)
        y1_predict = out[:, 0]
        y2_predict = out[:, 1]
        loss1 = loss_fn(y1_predict[data.train_mask], data.y[data.train_mask][:,0])#lon
        loss2 = loss_fn(y2_predict[data.train_mask], data.y[data.train_mask][:,1])
        loss = loss1+loss2
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        temp_y1_predict = y1_predict.detach().cpu().numpy()
        temp_y2_predict = y2_predict.detach().cpu().numpy()
        old_y1_predict = target_scaler2.inverse_transform(temp_y1_predict.reshape(-1, 1))#lon
        old_y2_predict = target_scaler1.inverse_transform(temp_y2_predict.reshape(-1, 1))
        with torch.no_grad():
            if use_cuda:
                model.cpu()

            sum_dis = 0
            dis_list = []
            for i, temp in enumerate(data.y[data.val_mask]):
                #
                real_lon_label = temp[0]
                real_lat_label = temp[1]
                #
                temp_lon_pre = old_y1_predict[val_node_id_list][i][0]
                temp_lat_pre = old_y2_predict[val_node_id_list][i][0]
                temp_dis = geodistance(real_lon_label,real_lat_label , temp_lon_pre, temp_lat_pre)

                sum_dis += temp_dis
                dis_list.append(temp_dis)
            error_thisepoch = sum_dis / len(dis_list)
            median_error = mediannum(dis_list)
            if args.print_flag > 0:
                print(error_thisepoch)
            perf_str = 'ValEpoch:%d:avg_error:%.5f:median_error:%.5f\n'%(epoch,error_thisepoch,median_error)
            fw1.write(perf_str)
            fw1.flush()

            cur_best_pre_0, stopping_step, should_stop = early_stopping(error_thisepoch, cur_best_pre_0,
                                                                        stopping_step, expected_order='dec',
                                                                        flag_step=500)
            ## ---TEST only for last time after finish tuning parameter

            # sum_dis = 0
            # dis_list = []
            # for i, temp in enumerate(data.y[data.test_mask]):
            #     # temp_dis = 0
            #     real_lon_label = temp[0]
            #     real_lat_label = temp[1]
            #     # 经纬度又出问题了
            #     temp_lon_pre = old_y1_predict[test_node_id_list][i][0]
            #     temp_lat_pre = old_y2_predict[test_node_id_list][i][0]
            #     temp_dis = geodistance(real_lon_label, real_lat_label, temp_lon_pre, temp_lat_pre)
            #
            #     sum_dis += temp_dis
            #     dis_list.append(temp_dis)
            # error_thisepoch = sum_dis / len(dis_list)
            # median_error = mediannum(dis_list)
            # if args.print_flag > 0:
            #     print(error_thisepoch)
            # perf_str = 'TestEpoch:%d:avg_error:%.5f:median_error:%.5f\n' % (epoch, error_thisepoch, median_error)
            # fw1.write(perf_str)
            # fw1.flush()
            # for temp in dis_list:
            #     fw1.write(str(temp) + ',')
            # fw1.write('\n')

            #----TEST


            if  use_cuda:
                model.cuda()
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break

