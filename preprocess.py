"""Data loading + preprocessing for NSL-KDD (shared by train.py and app.py)."""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# NSL-KDD: 41 features + attack label + difficulty score (no header in the raw files)
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
REMOTE = {"KDDTrain+.txt": "KDDTrain%2B.txt", "KDDTest+.txt": "KDDTest%2B.txt"}


def ensure_data(data_dir: Path) -> None:
    """Download NSL-KDD train/test files if they are not already in data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for local, remote in REMOTE.items():
        target = data_dir / local
        if target.exists():
            continue
        print(f"Downloading {local} ...")
        try:
            urllib.request.urlretrieve(BASE_URL + remote, target)
        except Exception as exc:  # offline / blocked network
            raise SystemExit(
                f"Could not download {local}: {exc}\n"
                f"Download KDDTrain+.txt and KDDTest+.txt manually (NSL-KDD, e.g. from "
                f"Kaggle or github.com/defcom17/NSL_KDD) and put them in '{data_dir}/'."
            )


def load_dataset(path: Path):
    """Load + clean one NSL-KDD file.

    Returns X (DataFrame of 41 features), y (0 = Normal, 1 = Attack), and the
    original attack-type labels (useful for analysis).
    """
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # --- cleaning ---------------------------------------------------------
    for c in CATEGORICAL + ["label"]:
        df[c] = df[c].astype(str).str.strip().str.lower().str.rstrip(".")
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    # (remaining NaNs are imputed inside the pipeline, so the same logic is
    #  applied automatically to anything a user uploads in the web app)

    y = (df["label"] != "normal").astype(int)
    return df[FEATURES].copy(), y, df["label"]


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values and encode categoricals.

    Ordinal encoding is compact and ideal for tree models. Categories never
    seen in training become -1 instead of crashing the app.
    """
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    num = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [("cat", cat, CATEGORICAL), ("num", num, NUMERIC)],
        verbose_feature_names_out=False,
    )
