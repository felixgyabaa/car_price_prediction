# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("dataset/clean_df.csv")

# %%
df.drop({df.columns[0], df.columns[1]}, axis=1, inplace=True)

# %%
print(df.dtypes)

# %%
df.corr(numeric_only=True)

# %%
df[["bore", "stroke", "compression-ratio", "horsepower"]].corr()

# %%
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(
    0,
)

# %%
df[["engine-size", "price"]].corr()

# %%
sns.regplot(x="highway-L/100km", y="price", data=df)

# %%
df[["highway-L/100km", "price"]].corr()

# %%
sns.regplot(x="stroke", y="price", data=df)

# %%
sns.boxplot(x="body-style", y="price", data=df)

# %%
sns.boxplot(x="engine-location", y="price", data=df)

# %%
sns.boxplot(x="drive-wheels", y="price", data=df)

# %%
df.describe(include=["object"])

# %%
df["drive-wheels"].value_counts()

# %%
drive_wheels_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheels_counts.rename(columns={"drive-wheels": "value_counts"}, inplace=True)

# %%
drive_wheels_counts

# %%
engine_loc_count = df["engine-location"].value_counts().to_frame()

# %%
engine_loc_count

# %%
df["drive-wheels"].unique()

# %%
df_group_1 = df[["drive-wheels", "body-style", "price"]]

# %%
df_group_1 = df_group_1.groupby(["drive-wheels"], as_index=False).mean(
    numeric_only=True
)
df_group_1

# %%
df_gptest = df[["drive-wheels", "body-style", "price"]]
grouped_test1 = df_gptest.groupby(["drive-wheels", "body-style"], as_index=False).mean()
grouped_test1

# %%
grouped_pivot = grouped_test1.pivot(index="drive-wheels", columns="body-style")
grouped_pivot

# %%
grouped_pivot = grouped_pivot.fillna(0)

# %%
sns.heatmap(grouped_pivot)

# %%
df.corr(numeric_only=True)

# %%
from scipy import stats

# %%
pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["length"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["width"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["curb-weight"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["engine-size"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["bore"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["city-L/100km"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
pearson_coef, p_value = stats.pearsonr(df["highway-L/100km"], df["price"])
print(f"Pearson Coefficient: {pearson_coef} \nP-Value: {p_value}")

# %%
grouped_test2 = df_gptest[["drive-wheels", "price"]].groupby(["drive-wheels"])
grouped_test2.head()

# %%
grouped_test2.get_group("4wd")["price"]

# %%
# ANOVA
f_val, p_val = stats.f_oneway(
    grouped_test2.get_group("4wd")["price"],
    grouped_test2.get_group("rwd")["price"],
    grouped_test2.get_group("fwd")["price"],
)

print(f"ANOVA Results: F-Value = {f_val}\tP-Value = {p_val}")

# %%
f_val, p_val = stats.f_oneway(
    grouped_test2.get_group("rwd")["price"],
    grouped_test2.get_group("fwd")["price"],
)

print(f"ANOVA Results: F-Value = {f_val}\tP-Value = {p_val}")

# %%
f_val, p_val = stats.f_oneway(
    grouped_test2.get_group("4wd")["price"],
    grouped_test2.get_group("fwd")["price"],
)

print(f"ANOVA Results: F-Value = {f_val}\tP-Value = {p_val}")
