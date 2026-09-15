# ---------------------------------------------------------------------------
# Feature-list reference for the nightly model validation (IMPROVEMENTS.md 3.4)
#
# WHY THIS FILE EXISTS
# --------------------
# The nightly job exists to catch a SILENT failure: if the model is ever
# retrained with the 26 features in a different ORDER, nothing raises. The
# model still loads, still predicts, and returns plausible-looking numbers -
# it just reads hour into day_of_week and so on. The only way to catch that is
# to compare the order against a reference that was captured when the model
# was known-good.
#
# The first version of the workflow tried to do that with:
#
#     assert list(cols) == list(cols), "feature ordering changed"
#
# which compares a list to itself and is therefore always true. It asserted
# nothing at all. This file is the reference that check should have used.
#
# HOW IT WAS PRODUCED (do not hand-edit)
# --------------------------------------
#     python -c "import joblib; print(joblib.load('models/feature_cols_clean.joblib'))"
#
# captured against models/best_rf_clean.joblib (1,127,963,617 bytes) whose
# own n_features_in_ == 26. If a retrain legitimately changes this order, the
# correct response is to update BOTH the model and this file in the same
# commit - so the change is deliberate and reviewable rather than silent.
#
# The order below is NOT sorted, and NOT the order features are computed in.
# It is the order the trained model was fitted on. Order is part of the
# contract (see IMPROVEMENTS.md item 2.9).
# ---------------------------------------------------------------------------

FEATURE_ORDER = [
    "hour",              # 0
    "day_of_week",       # 1
    "month",             # 2
    "year",              # 3
    "is_weekend",        # 4
    "hour_sin",          # 5
    "hour_cos",          # 6
    "day_of_week_sin",   # 7
    "day_of_week_cos",   # 8
    "month_sin",         # 9
    "month_cos",         # 10
    "load_lag_1h",       # 11
    "load_lag_24h",      # 12
    "load_lag_168h",     # 13
    "load_lag_720h",     # 14
    "load_roll_mean_24h",   # 15
    "load_roll_std_24h",    # 16
    "load_roll_mean_168h",  # 17
    "is_holiday",        # 18
    "is_month_start",    # 19
    "is_month_end",      # 20
    "load_roll_std_168h",   # 21
    "fourier_sin_1",     # 22
    "fourier_cos_1",     # 23
    "fourier_sin_2",     # 24
    "fourier_cos_2",     # 25
]

FEATURE_COUNT = len(FEATURE_ORDER)
