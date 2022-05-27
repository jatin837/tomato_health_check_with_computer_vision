"""This module provides basic utility functions"""
import tensorflow as tf


def get_train_test_val_split(train_size: float,
                         test_size: float,
                         val_size: float,
                         dataset: tf.data.Dataset) -> tuple[
                             tf.data.Dataset,
                             tf.data.Dataset,
                             tf.data.Dataset
                         ]:

    """return train, test, and validation split from tf.dataset"""
    size = len(dataset)
    train_size = int(train_size*size)
    test_size = int(test_size*size)
    val_size = int(val_size*size)

    train_ds = dataset.take(train_size)
    test_ds = dataset.skip(train_size).take(test_size)
    val_ds = dataset.skip(train_size).skip(val_size)

    return (train_ds, test_ds, val_ds)
