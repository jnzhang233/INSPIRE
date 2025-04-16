import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class Transformer_Agent(nn.Module):
    #需要的额外参数
    def __init__(self, input_shape, args):
        super(Transformer_Agent, self).__init__()
        self.args = args
        #参数设置，这部分参数需要在yaml文件定义，这里也会给默认值
        self.embedding_dim = args.transformer_embedding_dim if hasattr(args, 'transformer_embedding_dim') else args.rnn_hidden_dim
        self.n_head = args.transformer_n_head if hasattr(args, 'transformer_n_head') else 1
        self.n_layers = args.transformer_n_layers if hasattr(args, 'transformer_n_layers') else 1
        self.dropout = args.transformer_dropout if hasattr(args, 'transformer_dropout') else 0.0
        #和其他agent网络保持一致的参数
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_actions = args.n_actions

        #embedding生成层，作为输入层
        self.fc_embedding = nn.Linear(input_shape, self.embedding_dim)

        #Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_head,
            dim_feedforward=self.rnn_hidden_dim * 4,  # 通常 Transformer 中间层维度是 hidden_dim 的 4 倍
            dropout=self.dropout,
            batch_first=True  # 设置 batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        #输出层，输出各动作的评分（Q函数值）
        self.fc2 = nn.Linear(self.embedding_dim, self.n_actions)


        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.embedding_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc_embedding)
            orthogonal_init_(self.fc2, gain=args.gain)
            for name, param in self.transformer_encoder.named_parameters():
                if 'weight' in name:
                    orthogonal_init_(param)

    def init_hidden(self):
        # Transformer不存在显式的隐藏状态，故此函数无效。为了防止高级模块调用错误，放一个空函数
        return None

    def forward(self, inputs,hidden_state=None):
        #正常的forward函数，接收一个时间步的数据inputs，输出可选动作评分。hidden_state的输入和输出单纯是为了保持调用接口一致，没有别的用
        #也是runner调用的版本

        #获取inputs的shape，格式是(batch_size, n_agents, feature_size)
        b, a, e = inputs.size()

        #生成embedding
        embedded_inputs = F.relu(self.fc_embedding(inputs), inplace=True) #(b, a, embedding_size)

        #导入transformer，采用n_agents作为序列
        transformer_output = self.transformer_encoder(embedded_inputs)

        #生成动作评分
        q = self.fc2(transformer_output) # 形状是 (b, a, n_actions)

        return q,hidden_state

    def forward_using_embedding(self,embedded_inputs):
        #learner会用到的forward函数，这个版本输入来自其他agent的embedding并生成动作评分

        #这里和forward的transformer部分一致，要修改就一起修改
        # 导入transformer，采用n_agents作为序列
        transformer_output = self.transformer_encoder(embedded_inputs)

        # 生成动作评分
        q = self.fc2(transformer_output)  # 形状是 (b, a, n_actions)

        return q

    def get_embedding(self,inputs):
        #learner会用到的函数，输入自己的inputs并输出embedding

        # 生成embedding
        embedded_inputs = F.relu(self.fc_embedding(inputs), inplace=True)  # (b, a, embedding_size)

        return embedded_inputs