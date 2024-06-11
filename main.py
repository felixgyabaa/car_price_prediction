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
df["city-L/100km"] = 235 / df["city-mpg"]

# %%
df.head()

# %%
df["highway-L/100km"] = 235 / df["highway-mpg"]

# %%
df.drop(["city-mpg", "highway-mpg"], axis=1, inplace=True)

# %%
df[["length", "width", "height"]].head()

# %%
df["height"] = df["height"] / df["height"].max()

# %%
df["width"] = df["width"] / df["width"].max()
df["length"] = df["length"] / df["length"].max()

# %%
df["horsepower"] = df["horsepower"].astype(int, copy=True)

# %%
import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# %%
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

# %%
group_names = ["Low", "Medium", "High"]

# %%
df["horsepower-binned"] = pd.cut(
    df["horsepower"], bins, labels=group_names, include_lowest=True
)

df[["horsepower", "horsepower-binned"]].head(20)

# %%
df["horsepower-binned"].value_counts()

# %%
pyplot.hist(df["horsepower-binned"], bins=3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# %%
