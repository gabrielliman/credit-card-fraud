"""Feature-engineering steps (exp05+). Each step is target-free, so it may use train+test together
(no label leakage). Steps are listed in config.json under "features"."""
import numpy as np
import pandas as pd

FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
LOW_CARD_NUM = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes",
                "time_of_day_hour", "day_of_week", "card_age_months"]
NO_SIGNAL = ["card_type", "channel", "device_type", "time_of_day_hour", "day_of_week",
             "distance_from_home_km", "account_balance_usd", "card_age_months"]  # EDA AUC ≈ 0.50


def velocity(all_):
    # (a) velocity_score is correlated with txn_count (rho .61) and hours (rho .44): split it into the part
    # those two explain (OLS) and the residual, plus an explicit rate feature.
    A = np.c_[np.ones(len(all_)), all_["txn_count_last_24h"], np.log1p(all_["hours_since_last_txn"])]
    coef, *_ = np.linalg.lstsq(A, all_["velocity_score"].to_numpy(), rcond=None)
    all_["velocity_resid"] = all_["velocity_score"] - A @ coef
    all_["txn_per_hour"] = all_["txn_count_last_24h"] / (all_["hours_since_last_txn"] + 0.01)
    return all_


def flag_count(all_):
    # (b) additive risk score: number of risk flags raised
    all_["n_flags"] = all_[FLAGS].sum(axis=1)
    return all_


def freq_enc(all_):
    # (c1) frequency of each categorical level
    for c in all_.select_dtypes(include="category").columns:
        all_[c + "_freq"] = all_[c].map(all_[c].value_counts(normalize=True)).astype(float)
    return all_


def lowcard_as_cat(all_):
    # (c2) few-valued numerics treated as unordered categories
    for c in LOW_CARD_NUM:
        all_[c] = all_[c].astype("category")
    return all_


def drop_no_signal(all_):
    # (d) remove features with no univariate signal
    return all_.drop(columns=NO_SIGNAL)


STEPS = dict(velocity=velocity, flag_count=flag_count, freq_enc=freq_enc,
             lowcard_as_cat=lowcard_as_cat, drop_no_signal=drop_no_signal)


def apply(steps, X, X_test):
    if not steps:
        return X, X_test
    all_ = pd.concat([X, X_test], keys=["tr", "te"])
    for s in steps:
        all_ = STEPS[s](all_)
    return all_.loc["tr"], all_.loc["te"]
