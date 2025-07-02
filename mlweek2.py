import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("/content/quikr_car.csv")

df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].astype(str).str.replace(',', '')
df['Price'] = df['Price'].astype(int)

df['kms_driven'] = df['kms_driven'].astype(str).str.replace(' kms', '').str.replace(',', '')
df = df[df['kms_driven'].notna()]
df = df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype(int)

df = df[df['fuel_type'].notna()]

le = LabelEncoder()
df['fuel_type'] = le.fit_transform(df['fuel_type'])
df['company'] = le.fit_transform(df['company'])

X = df[['company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: ₹{mae:.2f}")

comparison = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred.astype(int)})
print(comparison.head())

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()
