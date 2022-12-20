
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# from sklearn.datasets import load_boston
import pandas

!wget https://raw.githubusercontent.com/myoh0623/dataset/main/boston.csv
boston_df = pandas.read_csv("boston.csv")

del boston_df["Unnamed: 0"]

boston_df
# MEDV 이 학습시켜야 하는 Target Data 입니다.

# 통계량 확인
boston_df.describe()

boston_df

data = boston_df.iloc[:,:-1]
target = boston_df.iloc[:,-1:]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

"""### 보스턴 주택 가격에 대한 선형 회귀"""

model = LinearRegression(normalize = True)
model.fit(x_train, y_train)

"""- 회귀모델의 검증을 위한 또 다른 측정 지표 중 하나로 결정 계수(coefficient of determination, $R^2$) 사용"""

# 방법1
print(model.score(x_test, y_test))

# 방법2
from sklearn.metrics import r2_score
print(r2_score(y_test, model.predict(x_test)))

from sklearn.model_selection import cross_val_score

model = LinearRegression(normalize = True)
scores = cross_val_score(model, x_train, y_train, cv = 4, scoring="r2")

scores # R2 Score의 값
print(f"R2 score mean:{scores.mean()}, std:{scores.std()}")

model = LinearRegression(normalize = True)
scores = cross_val_score(model, x_train, y_train, cv = 4, scoring="neg_root_mean_squared_error")

scores # R2 Score의 값
print(f"neg_root_mean_squared_error score mean:{scores.mean()}, std:{scores.std()}")

"""생성된 회귀 모델에 대해서 평가를 위해 LinearRegression 객체에 포함된 두 개의 속성 값을 통해 수식을 표현
- intercept_: 추정된 상수항
- coef_: 추정된 가중치 벡터
"""

print('w_0 = ' + str(model.intercept_) + ' ')
for i, c in enumerate(model.coef_[0]):
  print('w_' + str(i+1) + " = " + str(c))

def plot_boston_price(expected, predicted):
  plt.figure(figsize=(8,4))
  plt.scatter(expected, predicted)
  plt.plot([5, 50], [5, 50], '--r') #기준점
  plt.xlabel('True price ($1,000s)')
  plt.ylabel('Predicted price ($1,000s)')
  plt.tight_layout()

predicted = model.predict(x_test) 
expected = y_test                 

plot_boston_price(expected, predicted)

"""# 캘리포니아 주택 가격 데이터

| 속성 | 설명 |
|------|------|
| MedInc | 블록의 중간 소득 |
| HouseAge | 블록의 중간 주택 연도 |
| AveRooms | 평균 방 수 |
| AveBedrms | 평균 침실 수 |
| Population | 블록 내 거주 중인 인구수 |
| AveOccup | 평균 주택 점유율 |
| Latitude | 주택 블록 위도 |
| Longitude | 주택 블록 경도 |
"""

from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

california.keys()

print(california.DESCR)

import pandas as pd

california_df = pd.DataFrame(california.data, columns = california.feature_names)
california_df["target"] = california.target

california_df.head()

import seaborn as sns
sns.pairplot(california_df.sample(5000)) #데이터가 많아서plot으로 그리기 어렵다.

"""### 캘리포니아 주택 가격에 대한 선형 회귀"""

california_df.plot(kind="scatter", x = "Longitude", y = "Latitude", figsize = (20, 15), alpha = 0.2,
c = "target", cmap = plt.get_cmap("ocean"), s = california_df["Population"]/100, label = "Population")

# !pip install folium

import folium

latitude = 37.88
longitude = -122.23

m = folium.Map(location=[latitude, longitude],
               zoom_start=17, 
               width=750, 
               height=500
              )


from folium.plugins import MarkerCluster


m = folium.Map(
    location=[latitude,longitude],
    zoom_start=15
)

coords = california_df[["Latitude","Longitude"]]
marker_cluster = MarkerCluster().add_to(m)
for lat, long in zip(coords['Latitude'], coords['Longitude']):
    folium.Marker([lat, long], icon = folium.Icon(color="green")).add_to(marker_cluster)
m

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(california.data, california.target, test_size = 0.2)

model.fit(x_train, y_train)

y_test_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

print("R2 Score", r2_score(y_test, y_test_predict))
print("MSE", mean_squared_error(y_test, y_test_predict))

y_train_predict = model.predict(x_train)
print("R2 Score", r2_score(y_train, y_train_predict))
print("MSE", mean_squared_error(y_train, y_train_predict))

def plot_california_prices(expected, predicted):
  plt.figure(figsize = (8, 4))
  plt.scatter(expected, predicted)
  plt.plot([0, 5], [0, 5], '--r')
  plt.xlabel('True price ($100,000s)')
  plt.ylabel('Predicted pirce ($100,000s)')
  plt.tight_layout()

predicted = model.predict(x_test)
expected = y_test

plot_california_prices(expected, predicted)

