from __future__ import annotations

from utils import engineer_features
from utils import engineer_features_39
from utils import engineer_features_158plus39


def build_feature_table(df, feature_set: str):
    feature_set = str(feature_set or '').strip()
    if feature_set == '39':
        return engineer_features_39(df)
    if feature_set == '158+39':
        return engineer_features_158plus39(df)
    return engineer_features(df)
