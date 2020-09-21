# Source - http://mccormickml.com/2020/07/29/smart-batching-tutorial/
import random

import tensorflow as tf  # type: ignore
from tqdm import tqdm
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
        self.logger.info("Encoding Data")
        all_input_ids = []
        for sentence in tqdm(sentences):
            input_ids = self.tokenizer.encode(
                text=sentence,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding=False,
            )
            all_input_ids.append(input_ids)
        return all_input_ids

    def get_batches(self, input_ids, labels):
        self.logger.info("Creating Batches")
        # sort by input_id length
        samples = sorted(zip(input_ids, labels), key=lambda x: len(x[0]))
        batch_ordered_sentences = []
        batch_ordered_labels = []

        # progress bar
        n = (len(samples) // self.batch_size) + 1
        pbar = tqdm(total=n)

        while len(samples) > 0:
            upd_batch_size = min(self.batch_size, len(samples))
            # pick a random index in the list of remaining samples to start our batch at.
            systemRandom = random.SystemRandom()
            select = systemRandom.randint(0, len(samples) - upd_batch_size)
            batch = samples[select : (select + upd_batch_size)]
            batch_ordered_sentences.append([s[0] for s in batch])
            batch_ordered_labels.append([s[1] for s in batch])
            del samples[select : select + upd_batch_size]
            pbar.update(1)
        pbar.close()
        return batch_ordered_sentences, batch_ordered_labels

    def add_padding_batchwise(self, batch_ordered_sentences, batch_ordered_labels):
        self.logger.info("Padding Sequences")
        inputs = []
        attn_masks = []
        labels = []

        for (batch_inputs, batch_labels) in tqdm(
            list(zip(batch_ordered_sentences, batch_ordered_labels))
        ):
            batch_padded_inputs = []
            batch_attn_masks = []
            # find the longest sample in the batch.
            max_size = max(len(sen) for sen in batch_inputs)
            for sen in batch_inputs:
                # num of tokens to pad
                num_pads = max_size - len(sen)
                # add `num_pads` padding tokens to the end of the sequence.
                padded_input = sen + [self.tokenizer.pad_token_id] * num_pads
                # attention mask: {1: real token, 0: padding token}
                attn_mask = [1] * len(sen) + [0] * num_pads
                batch_padded_inputs.append(padded_input)
                batch_attn_masks.append(attn_mask)

            inputs.append(tf.convert_to_tensor(batch_padded_inputs))
            attn_masks.append(tf.convert_to_tensor(batch_attn_masks))
            labels.append(tf.convert_to_tensor(batch_labels))
        return inputs, attn_masks, labels

    def create(self, sentences, labels):
        input_ids = self.encode_data(sentences)
        batch_ordered_sentences, batch_ordered_labels = self.get_batches(
            input_ids, labels
        )
        inputs, attn_masks, labels = self.add_padding_batchwise(
            batch_ordered_sentences, batch_ordered_labels
        )

        self.logger.info(f"Input IDs: {len(inputs)}")
        self.logger.info(f"Attention Mask: {len(attn_masks)}")
        self.logger.info(f"Labels: {len(labels)}")

        self.logger.info("Batch Sizes")
        for idx, x in enumerate(inputs):
            print(f"Batch{idx}: {x.shape}")

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
