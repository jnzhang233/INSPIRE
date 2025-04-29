import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class Transformer_Scorer(nn.Module):
    #需要的额外参数
    def __init__(self, input_shape,n_agents, args):
        #input_shape:(seq_len, n_agents, feature_size)
        super(Transformer_Scorer, self).__init__()
        self.args = args
        #参数设置，这部分参数需要在yaml文件定义，这里也会给默认值
        self.embedding_dim = args.scorer_embedding_dim if hasattr(args, 'scorer_embedding_dim') else args.rnn_hidden_dim
        self.n_head = args.scorer_n_head if hasattr(args, 'scorer_n_head') else 1
        self.n_layers = args.scorer_n_layers if hasattr(args, 'scorer_n_layers') else 1
        self.dropout = args.scorer_dropout if hasattr(args, 'scorer_dropout') else 0.0
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_agents = n_agents

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
        self.fc2 = nn.Linear(self.embedding_dim, self.n_agents)


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

    def forward(self, inputs):
        #正常的forward函数，接收一个seq的数据inputs，输出seq的动作评分。

        #获取inputs的shape，格式是(seq_len, n_agents, feature_size)
        batch_size, seq_len, num_agents, input_dim = inputs.size()

        #处理输入格式，按照seq输入
        inputs = inputs.view(batch_size, seq_len * num_agents, input_dim)

        #生成embedding
        embedded_inputs = F.relu(self.fc_embedding(inputs), inplace=True)

        #回复形状，修改为按照seq和agent输入
        embedded_inputs = embedded_inputs.view(seq_len * num_agents, batch_size, self.embedding_dim) #(batch_size, seq_len, num_agents, input_dim)

        #导入transformer，采用n_agents作为序列
        transformer_output = self.transformer_encoder(embedded_inputs)

        #这里只需要最后一个时间步的输出即可，
        transformer_output = transformer_output.view(seq_len, num_agents, batch_size, self.embedding_dim)
        transformer_output = transformer_output[-1,:,:,:]

        #生成评分矩阵
        score = self.fc2(transformer_output)
        score = score.view(batch_size,num_agents,num_agents)
        score = F.softmax(score,dim=-1)

        return score