from pathlib import Path
from typing import List, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from attrdict import AttrDict  # type: ignore
from pandas import DataFrame
from tensorflow.keras.utils import to_categorical  # type: ignore

from .logger import singleton_logger
from .metrics import matthews_correlation
from .smart_batching_dataset import BertDataset

logger = singleton_logger()


class ColaData:
    @staticmethod
    def describe_df(df: DataFrame, label: str) -> None:
        print(
            f"{label}\nShape: {df.shape}\nDistribution:\n{df['label'].value_counts()}\n"
        )

    @staticmethod
    def get_cola_xy(df: DataFrame) -> List[DataFrame]:
        return [df["sentence"], df["label"]]

    def sort_df_by_length(self, data: DataFrame) -> DataFrame:
        data["length"] = data["sentence"].str.len()
        data = data.sort_values(by="length", ascending=True).reset_index(drop=True)
        data = data.drop(columns=["length"])
        return data

    def get_cola_df(self):
        in_domain_train = pd.read_csv(
            self.path / "in_domain_train.tsv", sep="\t", names=self.cols
        )
        in_domain_val = pd.read_csv(
            self.path / "in_domain_dev.tsv", sep="\t", names=self.cols
        )
        out_domain_val = pd.read_csv(
            self.path / "out_of_domain_dev.tsv", sep="\t", names=self.cols
        )
        val = in_domain_val.append(out_domain_val)
        test = pd.read_csv(self.path / "../../../cola_in_domain_test.tsv", sep="\t")
        return [in_domain_train, val, test]

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.cols = ["source", "label", "notes", "sentence"]
        self.traindf, self.valdf, self.testdf = self.get_cola_df()

        self.traindf = self.sort_df_by_length(self.traindf)
        self.valdf = self.sort_df_by_length(self.valdf)

        self.x_train, self.y_train = self.get_cola_xy(self.traindf)
        self.x_val, self.y_val = self.get_cola_xy(self.valdf)

        self.y_train_enc = to_categorical(self.y_train)
        self.y_val_enc = to_categorical(self.y_val)

        logger.info(f"\nColaData instantiated from path: {self.path}\n")
        self.describe_df(self.traindf, "Train Data")
        self.describe_df(self.valdf, "Val Data")

    def train(self, config: AttrDict, model, loss_fn, optimizer):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        train_data_obj = BertDataset(
            config.max_len, config.model_name, config.train_batch_size
        )
        train_dataset = train_data_obj.create(self.x_train.values, self.y_train_enc)

        valid_data_obj = BertDataset(
            config.max_len, config.model_name, config.eval_batch_size
        )
        valid_dataset = valid_data_obj.create(self.x_val.values, self.y_val_enc)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=["accuracy", matthews_correlation],
        )

        self.model.fit(
            train_dataset, epochs=config.epochs, validation_data=valid_dataset
        )
        return self

    def create_submission(self):
        test_data_obj = BertDataset(
            self.config.max_len, self.config.model_name, self.config.eval_batch_size
        )
        test_dataset = test_data_obj.create(
            self.testdf["Sentence"].values, [[0, 1]] * len(self.testdf)
        )
        preds = self.model.predict(test_dataset)
        self.testdf["Label"] = np.argmax(preds, axis=1)
        logger.info(f"\n\nTest Data: \n{self.testdf['Label'].value_counts()}")
        self.testdf[["Id", "Label"]].to_csv("sample_submission.csv", index=False)
