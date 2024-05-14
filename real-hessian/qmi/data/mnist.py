# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/7/10
from typing import Tuple

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from numpy import asarray


def get_mnist_datasets() -> Tuple[Dataset, Dataset]:
    mnist_dataset = load_dataset("mnist")
    train_dataset = mnist_dataset["train"]
    test_dataset = mnist_dataset["test"]

    train_dataset = train_dataset.map(
        lambda x: {"pixel_values": asarray(x["image"]).reshape(1, 28, 28), "labels": x["label"]},
        remove_columns=["image", "label"],
    )
    test_dataset = test_dataset.map(
        lambda x: {"pixel_values": asarray(x["image"]).reshape(1, 28, 28), "labels": x["label"]},
        remove_columns=["image", "label"],
    )

    return train_dataset, test_dataset
