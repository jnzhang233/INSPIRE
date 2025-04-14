REGISTRY = {}

from .gnn_rnn_agent import GnnRNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_agent import RNNAgent
from .trnasformer_agent import Transformer_Agent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["gnn_rnn"] = GnnRNNAgent
REGISTRY["transformer"] = Transformer_Agent