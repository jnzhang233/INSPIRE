from .q_learner import QLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner
from .ppo_learner import PPOLearner
from .coma_learner import COMALearner
from .qmix_differ import QDivedeLearner
from .vdn_differ import VDN_Differ_Learner
from .qmix_newdiffer_test import INSPIRE_Learner
from .qmix_newdiffer__ESR import INSPIRE_Learner as INSPIRE_Learner_v0
from .qmix_newdiffer_ESRv1 import INSPIRE_Learner as INSPIRE_Learner_v1
from .qmix_super import SUPER_Learner
from .qmix_per import PER_Learner
from .QMIX_Kalei import Kalei_NQLearner
from .VDN_Kalei import Kalei_VDNLearner
from .ices_nq_learner import ICESNQLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["q_learner"] = QLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["coma_learner"] = COMALearner

#differ
REGISTRY["qmix_differ"] = QDivedeLearner
REGISTRY["vdn_differ"] = VDN_Differ_Learner

#code test
REGISTRY["inspire_learner"] = INSPIRE_Learner
REGISTRY["inspire_learner_v0"] = INSPIRE_Learner_v0
REGISTRY["inspire_learner_v1"] = INSPIRE_Learner_v1
#super_qmix
REGISTRY["super_learner"] = SUPER_Learner
#per_qmix
REGISTRY["per_learner"] = PER_Learner

#kalei
REGISTRY["Kalei_nq_learner"] = Kalei_NQLearner
REGISTRY["Kalei_vdn_learner"] = Kalei_VDNLearner

#ICES
REGISTRY["ices_nq_learner"] = ICESNQLearner