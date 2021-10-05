# Lab 2

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


# Familirización con los datos

df_master = pd.read_csv("train.csv")
display(df_master.head())
display(df_master.describe())

# A partir de lo anterior se puede notar que:
# 1. Las variables Date y StateHoliday no aparecen en el *describe()*, esto se debe a que no son variables numéricas.
# 4. El 83% de las ventas se realizan con la tienda abierta, y el 38.1% con promociones.
# 5. El 17.9% de las ventas se realizan en feriados escolares.

# Los estadísticos asociados a las variables Store y DayOfWeek no tienen mayor interpretación, dado que son categóricas.

# La lógica indica que StateHoliday debiera ser una variable binaria o categórica. Analicemos los tipos de datos de cada columna.

display(df_master.dtypes)

# Efectivamente StateHoliday no es una variable binaria como dice la lógica. A continuación se muestran los valores que toman las distintas columnas y la cantidad de apariciones

for columna in df_master.columns:
    print("\n\nColumn name: ", columna, "\nValue\t# Registers")
    display(df_master[columna].value_counts())

# A partir de lo anterior, se puede concluir que:
# 1. La mayoría de las ventas ocurren cuando la tienda está abierta.
# 2. Un gran porcentaje de ventas se hacen durante promociones
# 3. La variable StateHoliday tiene un comportamiento binario, pero está representada como un objeto.
# 4. La variable StateHoliday posee dos interpretaciones del valor 0, una de tipo *int* y la otra de tipo *string*.

# En base a esto se plantea el siguiente resumen:

# | Variable      |	Tipo          | Descripción |
# | :-:           | :-:           | :-:         |
# | Store         |    Categórica |  Identificador de la tienda       |
# | DayOfWeek     |    Categórica |  Identificador del día de semana       |
# | Date          |    Fecha      |  Fecha de la venta       |
# | Sales         |    Entera     |  Ventas       |
# | Customers     |    Entera     |  Clientes       |
# | Open          |    Binaria    |  1 si la tienda estaba abierta, 0 si no       |
# | Promo         |    Binaria    |  1 si habia promoción, 0 si no       |
# | StateHoliday  |    Categórica    |  a, b o c si era feriado estatal, 0 si no       |
# | SchoolHoliday |    Binaria    |  1 si era feriado escolar, 0 si no       |

df_master["StateHoliday"] = df_master["StateHoliday"].replace([0], "0")
print("\n\nColumn name: ", "StateHoliday", "\nValue\t# Registers")
display(df_master["StateHoliday"].value_counts())

# Datos faltantes

print("Datos faltantes por columna:")
display(df_master.shape[0] - df_master[~df_master["Store"].isna()].count())

# No tenemos datos faltantes. Las tiendas y sus registros estan ordenados por par `(Date DESC, Store ASC)`.

# Depuración
# Formato de columnas

transform = {"0": 0, "a": 1, "b": 2, "c": 3}
df_master["StateHoliday"] = df_master["StateHoliday"].map(
    lambda x: transform[x], na_action="ignore")
df_master["SchoolHoliday"] = df_master["SchoolHoliday"].replace(["0"], 0)
df_master["SchoolHoliday"] = df_master["SchoolHoliday"].replace(["1"], 1)

# Análisis de outliers

# No tiene sentido analizar *outliers* en las siguientes columnas:
# 1. Store (debido a que es un ID)
# 2. DayOfWeek (días de la semana se mueven de 1 a 7)
# 3. Open, Promo, SchoolHoliday (son binarias)
# 4. StateHoliday (es categórica)

# Por lo tanto, solo quedan 2 variables (Sales y Customers).

df_master["Customers"].hist(bins=100, figsize=(10, 6))
plt.title("Histograma clientes", fontsize=24)
plt.xlabel("Clientes", fontsize=18)
plt.ylabel("Cantidad de registros", fontsize=18)
print("Cantidad de registros sin ventas: ",
      df_master.value_counts("Customers", 0)[0])

df_master.boxplot(column="Customers", figsize=(10, 6))

# Veamos su relación respecto a las ventas

pd.DataFrame(df_master["Sales"] / df_master["Customers"]).hist(bins=100)
plt.title("Histograma ventas por cliente", fontsize=18)
plt.xlabel("Ventas por cliente", fontsize=14)
plt.ylabel("Frecuencia", fontsize=14)

# ventas por cliente
pd.DataFrame(df_master["Sales"] / df_master["Customers"]).boxplot()
plt.title("Boxplot ventas por cliente", fontsize=18)
plt.ylabel("Ventas por cliente", fontsize=14)

# Para corregir este comportamiento, apliquemos logaritmo sobre esta columna

df_master["Customers_log"] = df_master.apply(
    lambda row: 0 if row["Customers"] == 0 else np.log(row["Customers"]), axis=1)
df_master["Sales_log"] = df_master.apply(
    lambda row: 0 if row["Sales"] == 0 else np.log(row["Sales"]), axis=1)
# Se excluyen los ceros porque se indefine el logaritmo

print(df_master.head())

# Misión 1: entrenamiento de modelos

# Mis tres regresores son:
# - Lineal
# - SVM
# - Red neuronal

# Preparando el entorno

# Categorizar los valores de las columnas categóricas.

label_encoder = LabelEncoder()
for col in ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]:
    df_master[col] = label_encoder.fit_transform(df_master[col])

print(df_master.head())

# Normalizamos los valores de las columnas enteras (sin considerar al *target*, que en este caso es *Sales_log*).

# Para normalizar la variable *customers* se requiere separar en train y test. Normalizar en base a todo el *df* no es correcto, porque los datos de test son desconocidos al momento de entrenar.


def escalar(train, test, col):
    scaler = StandardScaler()
    # la escala se ajusta sólo para train
    train[col] = scaler.fit_transform(train[col])
    # y luego se aplica para test, que es desconocido
    test[col] = scaler.transform(test[col])
    return train, test


cut_33p = df_master.shape[0] // 3  # Para separar en entrenamiento y test

df = df_master.copy()  # copiamos el data frame para no seguir cambiando el original

# tercio más reciente para test (esto es válido solo porque están ordenados)
df_test = df.loc[:cut_33p].copy()
df_train = df.loc[(cut_33p+1):].copy()

col = ["Customers_log"]  # La feature a normalizar
df_train, df_test = escalar(df_train, df_test, col)

display(df_test.shape)
display(df_train.shape)

# Definimos las features y el target
features = ["Store", "DayOfWeek", "Date", "Customers_log",
            "Open", "Promo", "StateHoliday", "SchoolHoliday"]
target = "Sales_log"


def train_and_test_model(model, train, test, *args, **kwargs):
    d = {}
    d["reg"] = model(*args, **kwargs)
    d["reg"].fit(train[features], train[target])
    d["predictions"] = d["reg"].predict(test[features])
    d["selfpredict"] = d["reg"].predict(train[features])
    d["mse"] = metrics.mean_squared_error(test[target], d["predictions"])
    print("El error cuadrático medio es:", d["mse"])
    return d


# Regresión lineal

lineal = train_and_test_model(LinearRegression, df_train, df_test)
# CV
parameters_l = {"fit_intercept": [True, False], "normalize": [True, False]}
lineal_cv = train_and_test_model(
    GridSearchCV, df_train, df_test, LinearRegression(), parameters_l)


# Red neuronal (dura 2000 segundos o más)

mpl = train_and_test_model(MLPRegressor, df_train, df_test, max_iter=300)
# CV
parameters_mpl = {"max_iter": [300, 350, 400]}
mpl_cv = train_and_test_model(
    GridSearchCV, df_train, df_test, MLPRegressor(), parameters_mpl)

# Árbol de decisión (dura 200 segundos o más)

tree = train_and_test_model(DecisionTreeRegressor, df_train, df_test)
# CV
parameters_dtr = {"criterion": ["mse", "friedman_mse", "poisson"]}
mpl_cv = train_and_test_model(
    GridSearchCV, df_train, df_test, DecisionTreeRegressor(), parameters_dtr)


# Resumen

# Regresor|MSE|CV
# -|-|-
# Lineal|0,035|0,035
# Red neuronal|0,048|0,039
# Árbol de decisión|0,022|0,022

# Notas
# * Los valores de los MSE pueden variar al ejecutar nuevamente ya que hay aleatoriedad en la red neuronal y en el árbol de decisión, pero debieran ser similares.
# * No se encontró una variación en la version con *cross-over* de la regresion lineal dado que los parametros de este no permiten una variación en el entrenamiento del modelo.
# * El modelo de redes neuronales probo tardar mucho al probar múltiples variaciones de parametros.
# * Se utilizo 300 iteraciones como caso base para el modelo de redes neuronales ya que con menos iteraciones no alcanzaba a converger a una solución en la mayoria de los casos.

# El modelo que mejor ajusta es el árbol de decisión. Además, en este caso el *cross-validation* utilizado genera diferencias marginales, excepto por el modelo de redes neuronales que tuvo una mejora perceptible, por lo que parece que la configuración por *default* de los modelos es óptima para esta base de datos.

# Misión 2

df_store = pd.read_csv("store.csv")
display(df_store.head())
display(df_store.describe())

print(df_store.dtypes)

for columna in df_store.columns:
    print("\n\nColumn name: ", columna, "\nValue\t# Registers")
    display(df_store[columna].value_counts())

print("Datos faltantes por columna:")
display(df_store.shape[0] - df_store[~df_store["Store"].isna()].count())

columns_na = ["CompetitionDistance", "CompetitionOpenSinceMonth",
              "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]
for col in columns_na:
    df_store[col] = df_store[col].fillna(df_store[col].mean())
df_store["PromoInterval"] = df_store["PromoInterval"].fillna(
    df_store['PromoInterval'].value_counts().index[0])
display(df_store.shape[0] - df_store[~df_store["Store"].isna()].count())

# Assortment, StoreType, PromoInterval
transform_type = {"a": 0, "b": 1, "c": 2, "d": 3}
transform_assortment = {"a": 0, "b": 1, "c": 2}
transform_interval = {"Jan,Apr,Jul,Oct": 0,
                      "Feb,May,Aug,Nov": 1, "Mar,Jun,Sept,Dec": 2}
df_store["Assortment"] = df_store["Assortment"].map(
    lambda x: transform_assortment[x], na_action="ignore")
df_store["StoreType"] = df_store["StoreType"].map(
    lambda x: transform_type[x], na_action="ignore")
df_store["PromoInterval"] = df_store["PromoInterval"].map(
    lambda x: transform_interval[x], na_action="ignore")
df_store.dtypes

# Features categoricas
for col in ["Store", "StoreType", "Assortment", "PromoInterval", "Promo2"]:
    df_store[col] = label_encoder.fit_transform(df_store[col])

df = df_master.copy().merge(df_store, on="Store", how="left")
display(df.head())
display(df.describe())

cut_33p = df.shape[0] // 3  # Para separar en entrenamiento y test

df_2 = df.copy()  # copiamos el data frame para no seguir cambiando el original

# tercio más reciente para test (esto es válido solo porque están ordenados)
df_test = df_2.loc[:cut_33p].copy()
df_train = df_2.loc[(cut_33p+1):].copy()

col = ["Customers_log"]  # La feature a normalizar
df_train, df_test = escalar(df_train, df_test, col)
col = ["CompetitionDistance"]  # La feature a normalizar
df_train, df_test = escalar(df_train, df_test, col)

features = ["Store", "DayOfWeek", "Date", "Customers_log", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
            "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]
target = "Sales_log"

# Regresión lineal

lineal = train_and_test_model(LinearRegression, df_train, df_test)
# CV
parameters_l = {"fit_intercept": [True, False], "normalize": [True, False]}
lineal_cv = train_and_test_model(
    GridSearchCV, df_train, df_test, LinearRegression(), parameters_l)

# Red neuronal (dura 3000 segundos aprox.)

mpl = train_and_test_model(MLPRegressor, df_train, df_test, max_iter=300)
# CV
parameters_mpl = {"max_iter": [300, 350, 400]}
mpl_cv = train_and_test_model(
    GridSearchCV, df_train, df_test, MLPRegressor(), parameters_mpl)

# Árbol de decisión (dura 400 segundos aprox.)

tree = train_and_test_model(DecisionTreeRegressor, df_train, df_test)
# CV
parameters_mpl = {"criterion": ["mse", "friedman_mse", "poisson"]}
tree_cv = train_and_test_model(
    GridSearchCV, df_train, df_test, DecisionTreeRegressor(), parameters_mpl)

# Resumen

# Regresor|MSE|CV
# -|-|-
# Lineal|0,028|0,028
# Red neuronal|0,178|0,024
# Árbol de decisión|0,009|0,009


# Notas
# * Los valores de los MSE pueden variar al ejecutar nuevamente ya que hay aleatoriedad en la red neuronal y en el árbol de decisión, pero debieran ser similares.
# * No se encontro una variacion en la version con *cross-over* de la regresión lineal dado que los parametros de este no permiten una variación en el entrenamiento del modelo.
# * El modelo de redes neuronales probo tardar mucho al probar multiples variaciones de parametros.
# * Se utilizo 300 iteraciones como caso base para el modelo de redes neuronales ya que con menos iteraciones no alcanzaba a converger a una solucion en la mayoria de los casos.

# El modelo que mejor ajusta es el árbol de decisián. Además, en este caso el *cross-validation* utlizado genera diferencias marginales, por lo que parece que la configuracián por *default* de los modelos es óptima para esta base de datos.

# Claramente el modelo de arbol de decicison se vio mas beneficiado por la nueva informacion, el modelo de regresion lineal tambien se vio beneficiado en un menor grado.

# El gran incremento de rendimiento en el modelo de árbol de decisión se puede deber a que la información agregada posee varias variables categóricas que son de gran aporte a este tipo de modelos.

# Misión 3

pca = PCA(n_components=2)
df_reduced = pd.DataFrame(pca.fit_transform(
    df.copy().loc[:, df.columns != "Sales_log"]), columns=["0", "1"])
df_reduced["Sales_log"] = df["Sales_log"]
display(df_reduced)

cut_33p = df_reduced.shape[0] // 3  # Para separar en entrenamiento y test

# copiamos el data frame para no seguir cambiando el original
df_reduced_2 = df_reduced.copy()

# tercio más reciente para test (esto es válido solo porque están ordenados)
df_test_pca = df_reduced_2.loc[:cut_33p].copy()
df_train_pca = df_reduced_2.loc[(cut_33p+1):].copy()
features = ["0", "1"]
target = "Sales_log"

# Regresión lineal

lineal_pca = train_and_test_model(LinearRegression, df_train_pca, df_test_pca)

# Red neuronal (dura 300 segundos aprox.)

mpl_pca = train_and_test_model(
    MLPRegressor, df_train_pca, df_test_pca, max_iter=300)

# Árbol de decisión

tree_pca = train_and_test_model(
    DecisionTreeRegressor, df_train_pca, df_test_pca)

# Resumen

# Regresor|MSE|PCA
# -|-|-
# Lineal|0,028|4,694
# Red neuronal|0,178|0,040
# Árbol de decisión|0,009|0,003


# Notas
# * Los valores de los MSE pueden variar al ejecutar nuevamente ya que hay aleatoriedad en la red neuronal y en el árbol de decisión, pero debieran ser similares.
# * Se utilizo 300 iteraciones como caso base para el modelo de redes neuronales ya que con menos iteraciones no alcanzaba a converger a una solucion en la mayoria de los casos.

# El reducir la dimensionalidad de los datos a 2 *features* tenemos una pérdida de rendimiento de varios grados de magnitud en la regresión lineal y una mejora importante en tanto la red neuronal y como en el árbol de decisión.

def plot_var_and_predictions(model_returned, test, title, predictions="predictions", variable="Sales_log"):
    plt.plot(np.exp(test[variable]), np.exp(model_returned[predictions]), ".")
    plt.ylabel("Predictions")
    plt.xlabel(variable)
    plt.title(title)
    plt.show()


plot_var_and_predictions(
    lineal, df_train, "Lineal (train)", predictions="selfpredict")
plot_var_and_predictions(lineal_pca, df_train_pca,
                         "Lineal PCA (train)", predictions="selfpredict")
plot_var_and_predictions(lineal, df_test, "Lineal (test)")
plot_var_and_predictions(lineal_pca, df_test_pca, "Lineal PCA (test)")

plot_var_and_predictions(
    mpl, df_train, "Red neuronal (train)", predictions="selfpredict")
plot_var_and_predictions(mpl_pca, df_train_pca,
                         "Red neuronal PCA (train)", predictions="selfpredict")
plot_var_and_predictions(mpl, df_test, "Red neuronal (test)")
plot_var_and_predictions(mpl_pca, df_test_pca, "Red neuronal PCA (test)")

plot_var_and_predictions(
    tree, df_train, "Árbol (train)", predictions="selfpredict")
plot_var_and_predictions(tree_pca, df_train_pca,
                         "Árbol PCA (train)", predictions="selfpredict")
plot_var_and_predictions(tree, df_test, "Árbol (test)")
plot_var_and_predictions(tree_pca, df_test_pca, "Árbol PCA (test)")

# Gráficamente podemos confirmar los resultados obtenidos. Las predicciones del modelo de regresión lineal están bastante alejadas del modelo sin reducción de dimensionalidad, luego vemos como la red neuronal se acerca bastante más al set de validación y finalmente tenemos el modelo de árbol de decisión que logra una precisión muy alta.
