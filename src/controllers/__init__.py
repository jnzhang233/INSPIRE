REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .transformer_controller import Transformer_MAC
from .Kalei_type_n_controller import Kalei_type_NMAC
from .ices_n_controller import ICESNMAC
from .ices_controller import ICESMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["transformer_mac"] = Transformer_MAC
REGISTRY["Kalei_type_n_mac"] = Kalei_type_NMAC
REGISTRY["ices_n_mac"] = ICESNMAC
REGISTRY["ices_mac"] = ICESMAC