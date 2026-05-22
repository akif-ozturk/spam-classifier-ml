import pandas as pd
import os

print("Çalışma klasörü:", os.getcwd())
print("Dosyalar:", os.listdir())

import pandas as pd

# dataset yükle
df = pd.read_csv("spam.csv", encoding="latin-1")

# gereksiz kolonları at
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# head
print(df.head())

# shape
print(df.shape)

# columns
print(df.columns)

# missing values
print(df.isnull().sum())

# duplicate
print(df.duplicated().sum())

# label dağılımı
print(df["label"].value_counts())

# label encode
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print(df.head())