from .run import run as default_run
from .per_run import run as per_run
from .ices_run import run as ices_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["per_run"] = per_run
REGISTRY["ices_run"] = ices_run