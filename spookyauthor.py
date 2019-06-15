#!/usr/bin/python

import pandas as pd

from collections import Counter

data = pd.read_csv("data/spooky-author-identification/train.csv")

authors = data.groupby(by='author').text.apply(" ".join)

result = (
    authors
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

eap = result["EAP"]
