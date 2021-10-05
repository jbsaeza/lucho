import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display


df = pd.read_csv("data.csv")
# display(df.head(10))
# display(df['Environmental_risk'].describe())


# Mision 1

# Ajustar datos extremos
# df["O3"].hist(bins=20)
# plt.show()
# df["O3_log"] = np.log(df["O3"])
# df["O3_log"].hist(bins=20)
# plt.show()

q1 = df["O3"].quantile(0.25)
q3 = df["O3"].quantile(0.75)
iqr = q3-q1
df = df[(df["O3"].empty) or (df["O3"] <= q3+1.5*iqr)]
df = df[(df["O3"].empty) or (df["O3"] >= q1-1.5*iqr)]

q1 = df["PM2.5"].quantile(0.25)
q3 = df["PM2.5"].quantile(0.75)
iqr = q3-q1
df = df[(df["PM2.5"].empty) or (df["PM2.5"] <= q3+1.5*iqr)]
df = df[(df["PM2.5"].empty) or (df["PM2.5"] >= q1-1.5*iqr)]


# Tratar datos faltantes

df_media = df.copy()
df_del = df.copy()

for header in df.columns:
    if header != "Environmental_risk":
        df_media[header].fillna(df_media[header].mean(), inplace=True)
    else:
        df_media[header].fillna("medio", inplace=True)
