import pandas as pd
import numpy as np
from IPython.display import display
import math

dt_cols = [
    "Arrival at Pickup - Time",
    "Confirmation - Time",
    "Pickup - Time",
    "Placement - Time",
]


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    res = [
        rmse(m.predict(X_train), y_train),
        rmse(m.predict(X_test), y_test),
        m.score(X_train, y_train),
        m.score(X_test, y_test),
    ]
    if hasattr(m, "oob_score_"):
        res.append(m.oob_score_)
    print(res)


def convert_dt(df, dt_cols):
    df[dt_cols] = df[dt_cols].apply(pd.to_datetime)
    return df


def create_submission(df_sample, df_test, model, name):
    y = model.predict(df_test)
    df_sample["Time from Pickup to Arrival"] = y
    df = df_sample.set_index("Order_No")
    df.to_csv(f"{name}.csv")
    return df


def create_submission_log(df_sample, df_test, model, name):
    y_mid = model.predict(df_test)
    y = np.exp(y_mid)
    df_sample["Time from Pickup to Arrival"] = y
    df = df_sample.set_index("Order_No")
    df.to_csv(f"{name}.csv")
    return df


def day(x):
    if (x >= 7 and x <= 9) or (x >= 16 and x <= 19):
        return "rush"
    else:
        return "normal"


def month(x):
    if x <= 10:
        return "beg_month"
    elif x > 10 and x <= 20:
        return "midmonth"
    else:
        return "endmonth"


def month_two(x):
    if x <= 15:
        return "beg_month"
    else:
        return "endmonth"


def rf_feat_importance(m, df):
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}
    ).sort_values("imp", ascending=False)


def plot_fi(fi):
    return fi.plot("cols", "imp", "barh", figsize=(12, 7), legend=False)

