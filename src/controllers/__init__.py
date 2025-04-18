REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .transformer_controller import Transformer_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["transformer_mac"] = Transformer_MAC