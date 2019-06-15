#!/usr/bin/python

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)
import pandas as pd

from collections import Counter

data = pd.read_csv("data/spooky-author-identification/train.csv").groupby(by='author')
d = dict(list(data))

result = (
    data.text.apply(" ".join)
        .str.lower()
        .str.replace(".", "")
        .str.replace(",", "")
        .str.replace("'", "")
        .str.replace("\"", "")
        .str.replace(",", "")
        .str.replace(";", "")
        .str.replace("!", "")
        .str.replace("?", "")
        .str.split()
        .apply(Counter)
)

cv = CountVectorizer()
tfid = TfidfTransformer(
    smooth_idf=False,
)

text = list(d["EAP"].text) + list(d["MWS"].text) + list(d["HPL"].text)

counts = cv.fit_transform(text)
tf_counts = tfid.fit_transform(counts)
