import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pandas.read_csv("./train_data.csv")
print(data)

data.plot.scatter(x='X',y='Y')
# plt.show()

features = data["X"].values.reshape(-1,1)
target = data["Y"]
regression = LinearRegression()
model = regression.fit(features,target)
print(model.coef_)
print(model.intercept_)