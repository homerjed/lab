from made import MADE
from batch import BatchNormFlow
from perm import Perm
from flow import Flow

data_dim = 2
H = 32
context_dim = 2

flow = Flow(
    MADE(data_dim, H, context_dim, act="tanh"),
    BatchNormFlow(data_dim),
    Perm(data_dim),
    MADE(data_dim, H, context_dim, act="tanh"),
    BatchNormFlow(data_dim),
    Perm(data_dim),
    MADE(data_dim, H, context_dim, act="tanh"),
)

