"""Training-fold resampling (D010). Never applied to validation or test data."""
import pandas as pd
from .config import SEED


def resample(method, X, y, cats):
    if not method:
        return X, y
    if method in ("smote", "smote_round"):
        # SMOTENC: SMOTE (Chawla et al., 2002) variant for mixed data. Categorical columns and 0/1 flags
        # take the most frequent value among the neighbours instead of being interpolated.
        from imblearn.over_sampling import SMOTENC
        flags = [c for c in X.columns if c not in cats and set(X[c].unique()) <= {0, 1}]
        cat_cols = cats + flags
        Xc = X.copy()
        for c in cats:
            Xc[c] = Xc[c].cat.codes
        sm = SMOTENC(categorical_features=[X.columns.get_loc(c) for c in cat_cols], random_state=SEED)
        Xr, yr = sm.fit_resample(Xc, y)
        for c in cats:
            Xr[c] = pd.Categorical.from_codes(Xr[c].astype(int), categories=X[c].cat.categories)
        if method == "smote_round":
            # D010 follow-up: round integer-valued columns back to integers so synthetic rows
            # can't be identified by non-integer values that real data never has.
            for c in X.columns:
                if c not in cat_cols and pd.api.types.is_integer_dtype(X[c]):
                    Xr[c] = Xr[c].round().astype(X[c].dtype)
        return Xr, yr
    raise ValueError(method)
