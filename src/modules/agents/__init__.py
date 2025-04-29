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
from .ices_grf_agent import ICESAgent as ICES_GRF_Agent
from .inspire_rnn_agent import INSPIRE_RNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["gnn_rnn"] = GnnRNNAgent
REGISTRY["transformer"] = Transformer_Agent

#inspire
REGISTRY["inspire"] = INSPIRE_RNNAgent

#kalei
REGISTRY["n_rnn_1R3"] = NRNNAgent_1R3
REGISTRY["Kalei_type_n_rnn_1R3"] = Kalei_type_NRNNAgent_1R3

#ices
REGISTRY["ices"] = ICESAgent
REGISTRY["ices_grf"] = ICES_GRF_Agent