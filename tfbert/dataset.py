import tensorflow as tf  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from .logger import singleton_logger


class BertDataset:
    def __init__(self, max_len, model_name, batch_size):
        self.max_len = max_len
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.logger = singleton_logger()

    def encode_data(self, sentences):
        encoded = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"]
        return input_ids

    def create(self, sentences, labels):
        input_ids = self.encode_data(sentences)
        self.logger.info(f"Input IDs: {len(input_ids)}")
        dataset = tf.data.Dataset.from_tensor_slices((input_ids, labels))
        dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
