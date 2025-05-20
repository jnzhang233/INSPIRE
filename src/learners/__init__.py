from .q_learner import QLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner
from .ppo_learner import PPOLearner
from .coma_learner import COMALearner
from .qmix_differ import QDivedeLearner
from .vdn_differ import VDN_Differ_Learner
from .qmix_newdiffer_final import INSPIRE_Learner as INSPIRE_Learner
from .qmix_newdiffer_final_grf import INSPIRE_Learner as INSPIRE_Learner_grf
from .qmix_newdiffer_other_priority import INSPIRE_Learner as INSPIRE_Learner_priority
from .qmix_super import SUPER_Learner
from .qmix_per import PER_Learner
from .QMIX_Kalei import Kalei_NQLearner
from .VDN_Kalei import Kalei_VDNLearner
from .ices_nq_learner import ICESNQLearner
from .ices_dmaq_qatten_learner import ICES_DMAQ_qattenLearner


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
REGISTRY["inspire_learner_grf"] = INSPIRE_Learner_grf
REGISTRY["inspire_learner_priority"] = INSPIRE_Learner_priority
#super_qmix
REGISTRY["super_learner"] = SUPER_Learner
#per_qmix
REGISTRY["per_learner"] = PER_Learner

#kalei
REGISTRY["Kalei_nq_learner"] = Kalei_NQLearner
REGISTRY["Kalei_vdn_learner"] = Kalei_VDNLearner

#ICES
REGISTRY["ices_nq_learner"] = ICESNQLearner
REGISTRY["ices_QPLEX"] = ICES_DMAQ_qattenLearner