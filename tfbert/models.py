import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

from transformers import TFAutoModel  # type: ignore


class BaseModel(Model):
    def __init__(self, model_name):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.base_model = TFAutoModel.from_pretrained(self.model_name)
        self.dropout = Dropout(0.3)
        self.dense = Dense(2, activation="softmax")

    @tf.function
    def call(self, inputs):
        # last_hidden_state -> (batch_size, sequence_length, hidden_size)
        last_hidden_state = self.base_model(inputs)[0]
        # cls_embeddings -> (batch_size, hidden_size)
        cls_embedd = last_hidden_state[:, 0, :]
        out_values = self.dropout(cls_embedd)
        out_values = self.dense(out_values)
        return out_values
