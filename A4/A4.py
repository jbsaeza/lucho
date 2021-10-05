import pandas as pd
import numpy as np
import datetime
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")
# display(df.head())
# print("Cantidad de columnas: {}".format(len(df.index)))
# display(df.describe())

# print(df["StateHoliday"].dropna().unique())
# df['Sales'].hist(bins=50)
# plt.show()
# df.boxplot(column='Sales')
# plt.show()
# df['Customers'].hist(bins=50)
# plt.show()
# df['Promo'].hist(bins=50)
# plt.show()
# print(df.apply(lambda x: sum(x.isnull()), axis=0))

# Completar NAs, Store esta en orden con maximo de 1115, Fecha esta en orden desc, dayofweek con datetime .weekday(), open if sales != 0 -> = 1, cutomers, promo y holidays drop
df = df[(df["Customers"].notna() & df["Promo"].notna()
        & df["StateHoliday"].notna() & df["SchoolHoliday"].notna())]
df['NextStore'] = df['Store'].shift(-1)
df['NextNextStore'] = df['Store'].shift(-2)
df['NNNStore'] = df['Store'].shift(-3)
df['PrevStore'] = df['Store'].shift(1)
df['PrevPrevStore'] = df['Store'].shift(2)
df['PPPStore'] = df['Store'].shift(3)
df['Store'] = np.where(df['Store'] > 0, df['Store'], df['NextStore'])
df['Store'] = np.where(df['Store'] > 0, df['Store'], df['PrevStore'])
df['Store'] = np.where(df['Store'] > 0, df['Store'],
                       df['NextNextStore'])
df['Store'] = np.where(df['Store'] > 0, df['Store'],
                       df['PrevPrevStore'])
df['Store'] = np.where(df['Store'] > 0, df['Store'],
                       df['NNNStore'])
df['Store'] = np.where(df['Store'] > 0, df['Store'],
                       df['PPPStore'])
df = df.drop(columns="NextStore")
df = df.drop(columns="PrevStore")
df = df.drop(columns="NextNextStore")
df = df.drop(columns="PrevPrevStore")
df = df.drop(columns="NNNStore")
df = df.drop(columns="PPPStore")

df['PrevDate'] = df['Date'].shift(1)
df['NextDate'] = df['Date'].shift(-1)
df['PrevPrevDate'] = df['Date'].shift(2)
df['NextNextDate'] = df['Date'].shift(-2)
df['PPPDate'] = df['Date'].shift(3)
df['NNNDate'] = df['Date'].shift(-3)
df['PPPPDate'] = df['Date'].shift(4)
df['NNNNDate'] = df['Date'].shift(-4)
df['PPPPPDate'] = df['Date'].shift(5)
df['NNNNNDate'] = df['Date'].shift(-5)
df['PPPPPPDate'] = df['Date'].shift(6)
df['NNNNNNDate'] = df['Date'].shift(-6)

df['Date'] = np.where(df['Date'].notna(), df['Date'], df['PrevDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['NextDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['PrevPrevDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['NextNextDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['PPPDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['NNNDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['PPPPDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['NNNNDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['PPPPPDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['NNNNNDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['PPPPPPDate'])
df['Date'] = np.where(df['Date'].notna(), df['Date'], df['NNNNNNDate'])

df = df.drop(columns="NextDate")
df = df.drop(columns="PrevDate")
df = df.drop(columns="NextNextDate")
df = df.drop(columns="PrevPrevDate")
df = df.drop(columns="NNNDate")
df = df.drop(columns="PPPDate")
df = df.drop(columns="NNNNDate")
df = df.drop(columns="PPPPDate")
df = df.drop(columns="NNNNNDate")
df = df.drop(columns="PPPPPDate")
df = df.drop(columns="NNNNNNDate")
df = df.drop(columns="PPPPPPDate")

for i in range(len(df)):
    row = df.iloc[i]
    x = row["Date"].split("-")
    df.iloc[i, 1] = datetime.datetime(
        int(x[0]), int(x[1]), int(x[2])).weekday() + 1

df['Open'] = np.where(df['Open'].notna(), df['Open'],
                      df['Sales'].notna().astype(float))

cat_vars = ['Store', "DayOfWeek", "Date",
            "Promo", "StateHoliday", "SchoolHoliday"]
label_encoder = LabelEncoder()
for i in cat_vars:
    df[i] = label_encoder.fit_transform(df[i])

test_set = df.iloc[:, :(df.shape[0]//3)]
training_set, test_set2 = train_test_split(df.copy(), test_size=0.3)
training_set, validation_set = train_test_split(
    training_set.copy(), test_size=0.1)

# scaler = StandardScaler()
features = ["Store", "DayOfWeek", "Date", "Customers",
            "Open", "Promo", "StateHoliday", "SchoolHoliday"]

# # training_set[features] = scaler.fit_transform(training_set[features])
# # validation_set[features] = scaler.transform(validation_set[features])
# # test_set[features] = scaler.transform(test_set[features])


def training_and_eval(model, training, eval, features, target):
    model.fit(training[features], training[target])
    predictions = model.predict(eval[features])
    mqe = metrics.mean_squared_error(predictions, eval[target])
    print(f"Error Cuadratico Medio: {mqe: .2}")


print("Test set tercio mas reciente")
target = 'Sales'
model = DecisionTreeClassifier()
training_and_eval(model, training_set, test_set, features, target)
model = LinearRegression()
training_and_eval(model, training_set, test_set, features, target)
model = KNeighborsClassifier()
training_and_eval(model, training_set, test_set, features, target)

print("Test set tercio aletorio")
model = DecisionTreeClassifier()
training_and_eval(model, training_set, test_set2, features, target)
model = LinearRegression()
training_and_eval(model, training_set, test_set2, features, target)
model = KNeighborsClassifier()
training_and_eval(model, training_set, test_set2, features, target)
