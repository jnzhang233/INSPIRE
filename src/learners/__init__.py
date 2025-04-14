from .q_learner import QLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner
from .ppo_learner import PPOLearner
from .coma_learner import COMALearner
from .q_learner_divide import QDivedeLearner
from .qmix_newdiffer_test import INSPIRE_Learner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["q_learner"] = QLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["coma_learner"] = COMALearner

#differ_qmix
REGISTRY["q_divide_learner"] = QDivedeLearner

#code test
REGISTRY["inspire_learner"] = INSPIRE_Learner