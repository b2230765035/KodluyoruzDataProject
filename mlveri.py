import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from veri import population

data=pd.read_csv("Cancer Deaths by Country and Type Dataset.csv")
population=pd.read_csv("population.csv")

population = population[population["Year"] >= 1990]
population = population.rename(columns={"Entity": "Country", "Population (historical)": "Population"})
population["Year"] = population["Year"].astype(str)

def hesapla_kanser_orani(df,nufus):
    grouped = df.groupby(["Country", "Year"]).sum(numeric_only=True)
    grouped["Total_Cancer"] = grouped.sum(axis=1)
    grouped = grouped.reset_index()
    grouped["Year"] = grouped["Year"].astype(str)
    merged = pd.merge(grouped, nufus, on=["Country", "Year"], how="left")
    merged = merged.dropna(subset=["Population"])
    merged["Cancer_per_100k"] = (merged["Total_Cancer"] / merged["Population"]) * 100000
    return merged

merged = hesapla_kanser_orani(data,population)
cancerYear=merged.groupby("Year")["Total_Cancer"].sum()
cancerCountry=merged.groupby("Country")["Total_Cancer"].sum().sort_values(ascending=False)

"""
# 1. Total_Cancer verisini al
df = merged.groupby("Year")["Total_Cancer"].sum().reset_index()
df["Year"] = df["Year"].astype(int)
df["Target"] = df["Total_Cancer"]

# 2. Geçmiş yılları özellik olarak ekle (lag features)
for i in range(1, 4):
    df[f"lag_{i}"] = df["Total_Cancer"].shift(i)

# 3. Geçmiş olmayan yılları at (NaN olan)
df_model = df.dropna().reset_index(drop=True)

# 4. X ve y ayır
X = df_model[["lag_1", "lag_2", "lag_3"]]
y = df_model["Target"]

# 5. Eğitim/Test ayır (2016 hedef olsun)
X_train = X[:-1]  # 1993–2015
y_train = y[:-1]
X_test = X[-1:]   # 2016'yı tahmin et
y_test = y[-1:]

# 6. Model kur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Tahmin et
y_pred = model.predict(X_test)

print("Gerçek 2016:", round(y_test.values[0]))
print("Tahmin 2016:", round(y_pred[0]))
print("RMSE:", round(np.sqrt(mean_squared_error([y_test.values[0]], y_pred)), 2))#54987.71



import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Zaten hazırlanmışsa yeniden gerek yok
# df: Year - Total_Cancer - lag_1 - lag_2 - lag_3
df = merged.groupby("Year")["Total_Cancer"].sum().reset_index()
df["Year"] = df["Year"].astype(int)
df["Target"] = df["Total_Cancer"]

# 2. Geçmiş 3 yılı özellik olarak ekle
for i in range(1, 4):
    df[f"lag_{i}"] = df["Total_Cancer"].shift(i)

df_model = df.dropna().reset_index(drop=True)

# 3. X ve y ayır
X = df_model[["lag_1", "lag_2", "lag_3"]]
y = df_model["Target"]

# 4. Eğitim: 1993–2015, Test: 2016
X_train = X[:-1]
y_train = y[:-1]
X_test = X[-1:]
y_test = y[-1:]

# 5. Model kur ve eğit
rf = RandomForestRegressor(n_estimators=100, random_state=42)#100 tane ağaç kullanır → Daha stabil sonuç
rf.fit(X_train, y_train)

# 6. Tahmin et
y_pred = rf.predict(X_test)

# 7. Sonuçları yazdır
print("🎯 Gerçek 2016:", round(y_test.values[0]))
print("📈 Tahmin 2016:", round(y_pred[0]))
print("📉 RMSE:", round(np.sqrt(mean_squared_error([y_test.values[0]], y_pred)), 2))#527959.92


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Toplam kanser verisi ve lag özellikleri
df = merged.groupby("Year")["Total_Cancer"].sum().reset_index()
df["Year"] = df["Year"].astype(int)
df["Target"] = df["Total_Cancer"]

# 2. Lag özellikleri ekle
for i in range(1, 4):
    df[f"lag_{i}"] = df["Total_Cancer"].shift(i)

df_model = df.dropna().reset_index(drop=True)

# 3. X ve y ayır
X = df_model[["lag_1", "lag_2", "lag_3"]]
y = df_model["Target"]

# 4. Eğitim ve test verisi (2016 test)
X_train = X[:-1]
y_train = y[:-1]
X_test = X[-1:]
y_test = y[-1:]

# 5. Model kur ve eğit
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# 6. Tahmin yap
y_pred = xgb.predict(X_test)

# 7. Sonuçlar
print("🎯 Gerçek 2016:", round(y_test.values[0]))
print("📈 Tahmin 2016:", round(y_pred[0]))
print("📉 RMSE:", round(np.sqrt(mean_squared_error([y_test.values[0]], y_pred)), 2))#RMSE: 367720.0


import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# 1. Veriyi hazırla
df = merged.groupby("Year")["Total_Cancer"].sum().reset_index()
df["Year"] = df["Year"].astype(int)
df["Target"] = df["Total_Cancer"]

# 2. Lag özellikleri
for i in range(1, 4):
    df[f"lag_{i}"] = df["Total_Cancer"].shift(i)

df_model = df.dropna().reset_index(drop=True)

# 3. X ve y ayır
X = df_model[["lag_1", "lag_2", "lag_3"]]
y = df_model["Target"]

# 4. Eğitim/Test ayır
X_train = X[:-1]
y_train = y[:-1]
X_test = X[-1:]
y_test = y[-1:]

# 5. LightGBM modeli kur
lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm.fit(X_train, y_train)

# 6. Tahmin yap
y_pred = lgbm.predict(X_test)

# 7. Sonuçları göster
print("🎯 Gerçek 2016:", round(y_test.values[0]))
print("📈 Tahmin 2016:", round(y_pred[0]))
print("📉 RMSE:", round(np.sqrt(mean_squared_error([y_test.values[0]], y_pred)), 2))#RMSE: 3051453.25
"""