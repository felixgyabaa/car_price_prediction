# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("dataset/clean_df.csv")
df

# %%
model = LinearRegression()
model

# %%
model.fit(df[["engine-size"]], df[["price"]])
model.predict(df[["engine-size"]])[0:5]

# %%
print(f"Coefficient: {model.coef_}\nIntercept: {model.intercept_}")

# %%
model1 = LinearRegression()
model1

# %%
model1.fit(df[["highway-mpg"]], df[["price"]])
model1.predict(df[["highway-mpg"]])[0:5]

# %%
print(f"Coefficient: {model1.coef_}\nIntercept: {model1.intercept_}")

# %%
model2 = LinearRegression()
model2

# %%
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
model2.fit(Z, df[["price"]])

# %%
print(f"Coefficiets: {model2.coef_}\nIntercept: {model2.intercept_}")

# %%
model3 = LinearRegression()
model3

# %%
model3.fit(df[["normalized-losses", "highway-mpg"]], df[["price"]])

# %%
print(f"Coefficiets: {model3.coef_}\nIntercept: {model3.intercept_}")

# %%
sns.regplot(x="highway-mpg", y="price", data=df)

# %%
sns.regplot(x="peak-rpm", y="price", data=df)


# %%
df[["peak-rpm", "highway-mpg", "price"]].corr()

# %%
sns.residplot(x=df[["highway-mpg"]], y=df[["price"]])

# %%
yhat = model2.predict(Z)

# %%
ax1 = sns.histplot(df["price"], color="r")
sns.histplot(yhat, color="b", ax=ax1)
plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")
plt.show()
plt.close()


# %%
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, ".", x_new, y_new, "-")
    plt.title("Polynomial Fit with Matplotlib for Price ~ Length")
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel("Price of Cars")

    plt.show()
    plt.close()


# %%
f = np.polyfit(df["highway-mpg"], df["price"], 3)
p = np.poly1d(f)
print(p)

# %%
PlotPolly(
    model=p,
    independent_variable=df["highway-mpg"],
    dependent_variabble=df["price"],
    Name="highway-mpg",
)

# %%
f1 = np.polyfit(df["highway-mpg"], df["price"], 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1, df["highway-mpg"], df["price"], "Highway MPG")

# %%
from sklearn.preprocessing import PolynomialFeatures

# %%
pr = PolynomialFeatures(degree=2)

# %%
Z_pr = pr.fit_transform(Z)

# %%
Z_pr.shape

# %%
Z.shape

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%
Input = [
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression()),
]

# %%
pipe = Pipeline(Input)
pipe

# %%
Z = Z.astype("float")
pipe.fit(Z, df["price"])

# %%
ypipe = pipe.predict(Z)
ypipe[0:4]

# %%
X = df[["engine-size"]]
Y = df[["price"]]

model.fit(X, Y)
print(f"The R-Squared Error is: {model.score(X,Y)}")

# %%
yhat = model.predict(X)

# %%
from sklearn.metrics import mean_squared_error

# %%
mse = mean_squared_error(df["price"], yhat)
print(f"The Mean Square Error is: {mse}")
# %%
