from tensorflow.keras.optimizers import Adam  # type: ignore

from config import configs_local  # type: ignore
from tfbert.cola_data import ColaData  # type: ignore
from tfbert.loss import bce  # type: ignore
from tfbert.models import BaseModel  # type: ignore

exp = ColaData(configs_local.datapath)
exp.train(
    configs_local,
    BaseModel(configs_local.model_name),
    bce,
    Adam(learning_rate=configs_local.lr),
)
exp.create_submission()
