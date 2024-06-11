# %%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# %%
url = "/Users/felixgyabaa/Documents/data_analysis/datasets/automobile/imports-85.data"

df = pd.read_csv(url, header=None)

# %%
df.head()
df.tail()

# %%
headers = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

df.columns = headers

# %%
path = "/Users/felixgyabaa/Documents/data_analysis/car_price_prediction/dataset/Automobile.csv"
df.to_csv(path)

# %%
df.replace("?", np.nan, inplace=True)

# %%
missing_data = df.isnull()

# %%
missing_data

# %%
for column in missing_data:
    print(f"{column} : {missing_data[column].value_counts()}")

# %%
df["normalized-losses"] = df["normalized-losses"].replace(
    np.nan, df["normalized-losses"].mean()
)

# %%
df.dtypes
# %%
df["normalized-losses"] = df["normalized-losses"].astype("float")

# %%
df["stroke"] = df["stroke"].astype("float")

# %%
df["stroke"] = df["stroke"].replace(np.nan, df["stroke"].mean())

# %%
df["bore"] = df["bore"].astype("float")

# %%
df["bore"] = df["bore"].replace(np.nan, df["bore"].mean())

# %%
df["horsepower"] = df["horsepower"].astype("float")

# %%
df["horsepower"] = df["horsepower"].replace(np.nan, df["horsepower"].mean())

# %%
df["peak-rpm"] = df["peak-rpm"].astype("float")

# %%
df["peak-rpm"] = df["peak-rpm"].replace(np.nan, df["peak-rpm"].mean())

# %%
mode = df["num-of-doors"].mode()
df["num-of-doors"] = df["num-of-doors"].replace(np.nan, mode[0])

# %%
df["price"].dropna(axis=0, inplace=True)

# %%
df.isnull().value_counts()

# %%
df.dropna(inplace=True)

# %%
df["normalized-losses"] = df["normalized-losses"].astype("int")

# %%
df[["bore", "stroke", "price", "peak-rpm"]] = df[
    ["bore", "stroke", "price", "peak-rpm"]
].astype("float")

# %%
df["horsepower"] = df["horsepower"].astype("object")
# %%
