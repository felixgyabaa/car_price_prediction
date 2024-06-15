# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("dataset/Automobile2.csv")
df.head()

# %%
df._get_numeric_data()

# %%
from ipywidgets import interact, interactive, fixed, interact_manual

# %%
import matplotlib.pyplot as plt
import seaborn as sns


# %%
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel("Price (in dollars)")
    plt.ylabel("Proportion of Cars")
    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, "ro", label="Training Data")
    plt.plot(xtest, y_test, "go", label="Test Data")
    plt.plot(
        x,
        lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))),
        label="Predicted Function",
    )
    plt.ylim([-10000, 60000])
    plt.ylabel("Price")
    plt.legend()


# %%
y_data = df["price"]
x_data = df.drop("price", axis=1)

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.4,
    random_state=0,
)

print(
    f"Number of test samples: {x_test.shape[0]}\nNumber of training samples: {x_train.shape[0]}"
)

# %%
from sklearn.linear_model import LinearRegression

model_lre = LinearRegression()
model_lre

# %%
model_lre.fit(x_train[["horsepower"]], y_train)

# %%
model_lre.score(x_test[["horsepower"]], y_test)

# %%
model_lre.score(x_train[["horsepower"]], y_train)

# %%
from sklearn.model_selection import cross_val_score

# %%
Rcross = cross_val_score(model_lre, x_data[["horsepower"]], y_data, cv=4)
Rcross

# %%
print(
    f"The means of the folds are: {Rcross.mean()} and the standard deviation is: {Rcross.std()}"
)

# %%
-1 * cross_val_score(
    model_lre, x_data[["horsepower"]], y_data, cv=4, scoring="neg_mean_squared_error"
)

# %%
from sklearn.model_selection import cross_val_predict

# %%
yhat = cross_val_predict(model_lre, x_data[["horsepower"]], y_data, cv=4)
yhat[0:5]

# %%
model_mlre = LinearRegression()
model_mlre.fit(
    x_train[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_train
)

# %%
yhat_train = model_mlre.predict(
    x_train[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
)
yhat_train[0:5]

# %%
yhat_test = model_mlre.predict(
    x_test[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
)
yhat_test[0:5]

# %%
Title = "Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution"
DistributionPlot(
    RedFunction=y_train,
    BlueFunction=yhat_train,
    RedName="Actual Values (Train)",
    BlueName="Predicted Values (Train)",
    Title=Title,
)

# %%
Title = "Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data"

DistributionPlot(
    RedFunction=y_test,
    BlueFunction=yhat_test,
    RedName="Actual Values (Test)",
    BlueName="Predicted Values (Test)",
    Title=Title,
)

# %%
from sklearn.preprocessing import PolynomialFeatures

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.45, random_state=0
)
# %%
x_train

# %%
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[["horsepower"]])
x_test_pr = pr.fit_transform(x_test[["horsepower"]])
pr

# %%
poly = LinearRegression()
poly.fit(x_train_pr, y_train)

# %%
yhat = poly.predict(x_test_pr)
yhat[0:5]

# %%
print(f"Predicted Values: {yhat[0:5]}\nTrue Values: {y_test[0:5].values}")

# %%
PollyPlot(x_train["horsepower"], x_test["horsepower"], y_train, y_test, poly, pr)

# %%
poly.score(x_train_pr, y_train)

# %%
poly.score(x_test_pr, y_test)

# %%
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[["horsepower"]])

    x_test_pr = pr.fit_transform(x_test[["horsepower"]])

    poly.fit(x_train_pr, y_train)

    Rsqu_test.append(poly.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel("order")
plt.ylabel("R^2")
plt.title("R^2 Using Test Data")
plt.text(3, 0.75, "Maximum R^2 ")


# %%
