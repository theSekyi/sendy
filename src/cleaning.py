import pandas as pd
import numpy as np
import math


def combine_train_test(train, test, rider, disjoint_cols):
    train["col_type"] = "train"
    test["col_type"] = "test"
    df_train_test = pd.concat(
        [train.drop(disjoint_cols, axis=1), test], ignore_index=True, sort=True
    )
    df_merged = df_train_test.merge(rider, how="left", on="Rider Id")
    df = df_merged.copy()
    return df.sort_values(by="orders")

