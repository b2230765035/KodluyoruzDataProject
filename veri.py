import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data=pd.read_csv('Cancer Deaths by Country and Type Dataset.csv')
#print(data.info())#28 veri Code de eksik
#print(data[data["Code"].isnull()]["Country"].unique())#iki ülkede eksik 28 veri
population=pd.read_csv("population.csv")
#boş satırları silelim
data.dropna(subset=["Code"],inplace=True)
#print(data)
#print(data["Code"].isnull().sum())  # 0 yazmalı
#print(data.columns[data.isnull().any()])#Tablondaki tüm sütunlar tam dolu.
data = data.reset_index(drop=True)
#print(data )
#print(data.info())

# Nüfus verisi düzenleme
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
#Her yıl tüm dünyadaki toplam kanser oranları ne kadar? Artıyor mu, azalıyor mu?
#Line plot yıl ekseninde değişimi daha iyi gösterir.

plt.figure(figsize=(10, 5))
merged.groupby("Year")["Total_Cancer"].sum().plot()
plt.title("Yıllara Göre Toplam Kanser Vakası (Dünya)")
plt.ylabel("Toplam Kanser Vakası")
plt.xlabel("Yıl")
plt.grid()
plt.tight_layout()
plt.show()


#Hangi ülkede toplamda daha fazla kanser var?
top10 = merged.groupby("Country")["Total_Cancer"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top10.plot(kind="barh")
plt.title("Toplam Kanser Vakası (En Yüksek 10 Ülke)")
plt.xlabel("Toplam Kanser Vakası")
plt.grid(axis='x')
plt.tight_layout()
plt.show()




# Her yıl için en yüksek kanser oranına sahip ilk 3 ülkeyi bul
top3_filtered = (
    merged.sort_values(["Year", "Cancer_per_100k"], ascending=[True, False])
    .groupby("Year")
    .head(3)
)

# Pivot tablo
pivot_df = top3_filtered.pivot(index="Year", columns="Country", values="Cancer_per_100k")

# Grafik
pivot_df.plot(kind="line", marker="o", figsize=(12, 6),
              title="Yıllara Göre En Fazla Kanser Oranına Sahip 3 Gerçek Ülke")
plt.ylabel("Toplam Kanser Oranı")
plt.xlabel("Yıl")
plt.grid(True)
plt.legend(title="Ülke")
plt.tight_layout()
plt.show()




#Her ülkenin ortalama yıllık kanser oranı ne? Genel ortalamadan ne kadar sapıyor?
#Yani sağlıkta iyileşme veya kötüleşme trendi analiz edilebilir

population["Country"] = population["Country"].str.strip().str.lower()
# data veri kümesindeki ülke adlarını da temizle
data["Country"] = data["Country"].str.strip().str.lower()
data["Year"] = data["Year"].astype(str)

# Yıla göre eşleşme için hem ülke hem yıl kullanılmalı
pop_lookup = population.set_index(["Country", "Year"])["Population"].to_dict()
data["Population"] = data.set_index(["Country", "Year"]).index.map(pop_lookup)

# Kanser sütunları
cancer_columns = data.columns[3:-1]

# Toplam ve normalize kanser
data["Total_Cancer"] = data[cancer_columns].sum(axis=1)
data["Cancer_per_100k"] = (data["Total_Cancer"] / data["Population"]) * 100000

# Yıl ve ülkeye göre toplam
grouped = data.groupby(["Year", "Country"]).sum(numeric_only=True)
grouped["Total"] = grouped["Total_Cancer"]

# Yıllık ortalama ve sapma
yearly_avg = grouped["Total"].groupby("Year").mean()
grouped = grouped.reset_index()
grouped["Sapma"] = grouped.apply(lambda row: row["Total"] - yearly_avg[row["Year"]], axis=1)

# Ortalama sapma ülke bazında
country_sapma_avg = grouped.groupby("Country")["Sapma"].mean()

# Kategorilere ayır
high = country_sapma_avg[country_sapma_avg > 500].index.tolist()
low = country_sapma_avg[country_sapma_avg < -500].index.tolist()
mid = country_sapma_avg[(country_sapma_avg >= -500) & (country_sapma_avg <= 500)].index.tolist()

# Grafik fonksiyonu
def draw_sapma_plot_for(countries, title):
    plt.figure(figsize=(12, 6))
    for country in countries:
        df = grouped[grouped["Country"] == country]
        plt.plot(df["Year"], df["Sapma"], label=country.title(), marker='o')
    plt.axhline(0, color="gray", linestyle="--")
    plt.title(title)
    plt.xlabel("Yıl")
    plt.ylabel("Toplam Kanser Sapması")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# Grafikleri çiz
draw_sapma_plot_for(high[:6], "🔴 Sürekli Ortalamanın Üzerinde Olan Ülkeler")
draw_sapma_plot_for(low[:6], "🟢 Sürekli Ortalamanın Altında Olan Ülkeler")
draw_sapma_plot_for(mid[:6], "🟡 Ortalama Etrafında Dalgalanan Ülkeler")

# Sütun isimlerinden boşlukları temizle
data.columns = data.columns.str.strip()
# Kanser sütunlarını seç (baştaki 3 metadata sütunu ve sondaki nüfus/normalize sütunları hariç)
cancer_columns = data.columns[3:-3]
cancer_columns = [col.strip() for col in cancer_columns]  # Boşlukları temizle

# Toplam kanser vakası ve normalize oran (100k kişi başına)
data["Total_Cancer"] = data[cancer_columns].sum(axis=1)
data["Cancer_per_100k"] = (data["Total_Cancer"] / data["Population"]) * 100000

# Her yıl kişi başına en yüksek kanser oranına sahip ülkeyi bul
top_country_per_year = (
    data.loc[data.groupby("Year")["Cancer_per_100k"].idxmax()]
    .reset_index(drop=True)
)

# Her yıl için o ülkenin en yaygın kanser türünü bul
most_common_types = []

for _, row in top_country_per_year.iterrows():
    year = row["Year"]
    country = row["Country"].strip().lower()

    row_data = data[(data["Year"] == year) & (data["Country"] == country)]

    if not row_data.empty:
        max_cancer_type = row_data[cancer_columns].iloc[0].idxmax()
        most_common_types.append(max_cancer_type)
    else:
        most_common_types.append("Bilinmiyor")

top_country_per_year["Most_Common_Cancer"] = most_common_types
top_country_per_year["Country"] = top_country_per_year["Country"].str.title()

# Grafik çizimi
plt.figure(figsize=(14, 6))
sns.barplot(data=top_country_per_year, x="Year", y="Cancer_per_100k", hue="Most_Common_Cancer", dodge=False)

# Her çubuğun üzerine ülke adını yaz
for i, row in top_country_per_year.iterrows():
    label = f"{row['Country']}"
    plt.text(x=i, y=row["Cancer_per_100k"] + 1, s=label, ha='center', fontsize=7, rotation=90)

plt.title("Yıllara Göre Kişi Başına En Yüksek Kanser Oranına Sahip Ülke ve Baskın Kanser Türü")
plt.ylabel("Kanser Oranı (100k Kişi Başına)")
plt.xlabel("Yıl")
plt.xticks(rotation=45)
plt.legend(title="Baskın Kanser Türü", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
continent_df = pd.read_csv("Countries by continents.csv")

merged["Country"] = merged["Country"].str.strip().str.lower()
continent_df["Country"] = continent_df["Country"].str.strip().str.lower()

merged_with_continent = pd.merge(merged, continent_df, on="Country", how="left")
cleaned = merged_with_continent.dropna(subset=["Continent"]).copy()

def yeni_risk_sinifi(total):
    if total < 1500:
        return 0
    elif total < 4000:
        return 1
    else:
        return 2

cleaned["Risk_Level"] = cleaned["Total_Cancer"].apply(yeni_risk_sinifi)

train_mask = cleaned["Year"].astype(int) <= 2014
test_mask = cleaned["Year"].astype(int) > 2014

X_geo = pd.concat([
    cleaned[["Population"]],
    pd.get_dummies(cleaned["Continent"], prefix="Continent")
], axis=1)
y_geo = cleaned["Risk_Level"]

X_train_g, X_test_g = X_geo[train_mask], X_geo[test_mask]

y_train_g, y_test_g = y_geo[train_mask], y_geo[test_mask]

scaler_g = StandardScaler()
X_train_g_scaled = scaler_g.fit_transform(X_train_g)
X_test_g_scaled = scaler_g.transform(X_test_g)

rf_geo = RandomForestClassifier(random_state=42)
rf_geo.fit(X_train_g_scaled, y_train_g)
y_pred_rf = rf_geo.predict(X_test_g_scaled)

print("🔍 Random Forest")
print("Accuracy:", accuracy_score(y_test_g, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test_g, y_pred_rf))
print("Classification Report:\n", classification_report(y_test_g, y_pred_rf))

##XGBOOST VE LİGHTGBM DENENDİ AMA EN BAŞARILI MODEL RANDOMFOREST OLARAK SAPTADIK

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_g_scaled, y_train_g)
y_pred_knn = knn.predict(X_test_g_scaled)

print("🔍 K-Nearest Neighbors")
print("Accuracy:", accuracy_score(y_test_g, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test_g, y_pred_knn))
print("Classification Report:\n", classification_report(y_test_g, y_pred_knn))

### K-Nearest Neighbors (KNN) Performansı
#KNN modeli genel olarak başarılı performans gösterdi, ancak sınıflar arasındaki sınırların belirgin olmadığı durumlarda karışıklık yaşandı.

from sklearn.tree import DecisionTreeClassifier

# Decision Tree modeli
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_g_scaled, y_train_g)
y_pred_dt = dt_model.predict(X_test_g_scaled)

print("🔍 Decision Tree")
print("Accuracy:", accuracy_score(y_test_g, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test_g, y_pred_knn))
print("Classification Report:\n", classification_report(y_test_g, y_pred_knn))

## Karar Ağacı Sınıflandırıcısı
#Karar ağacı modeli verileri iyi öğrenmiş görünmekte, fakat test setinde fazla genellemeye gidebildiği durumlar gözlemlendi.