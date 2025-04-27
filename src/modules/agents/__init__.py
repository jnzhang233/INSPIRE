REGISTRY = {}

from .gnn_rnn_agent import GnnRNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_agent import RNNAgent
from .transformer_agent import Transformer_Agent
from .Kalei_type_NRNNAgent import (
    NRNNAgent_1R3,
    Kalei_type_NRNNAgent_1R3,
)
from .ices_agent import ICESAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["gnn_rnn"] = GnnRNNAgent
REGISTRY["transformer"] = Transformer_Agent

#kalei
REGISTRY["n_rnn_1R3"] = NRNNAgent_1R3
REGISTRY["Kalei_type_n_rnn_1R3"] = Kalei_type_NRNNAgent_1R3

#ices
REGISTRY["ices"] = ICESAgent