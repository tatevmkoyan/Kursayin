# =====================================================================
# Տների գների կանխատեսում գծային ռեգրեսիայի (Linear Regression) ալգորիթմով
# Dataset՝ California Housing
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # օգտագործվում է միայն ստուգման համար

# ---------------------------------------------------------------------
# 1. ՏՎՅԱԼՆԵՐԻ ՀԱՎԱՔԱԾՈՒ
# ---------------------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Տվյալների չափը (տողեր, սյունակներ):", df.shape)
print(df.head(10))

# Սյունակների նկարագրություն
# MedInc     - շրջանի բնակիչների միջին եկամուտ
# HouseAge   - տան միջին տարիք
# AveRooms   - սենյակների միջին քանակ
# AveBedrms  - ննջասենյակների միջին քանակ
# Population - շրջանի բնակչություն
# AveOccup   - տանը բնակվողների միջին քանակ
# Latitude   - աշխարհագրական լայնություն
# Longitude  - աշխարհագրական երկայնություն
# MedHouseVal- տների միջին արժեք (թիրախ, հարյուր հազար $)

# ---------------------------------------------------------------------
# 2. ՏՎՅԱԼՆԵՐԻ ՆԱԽԱՊԱՏՐԱՍՏՈՒՄ
# ---------------------------------------------------------------------
print("\nԲաց թողնված արժեքներ ըստ սյունակների:")
print(df.isna().sum())
df = df.dropna()

print("\nՆկարագրական վիճակագրություն:")
print(df.describe().round(2))

X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

# Բաժանում՝ 80% ուսուցում, 20% թեստավորում
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} տող | Test: {X_test.shape[0]} տող")

# Ստանդարտացում (միայն train-ի միջինով և շեղումով, որպեսզի test-ից տվյալ չարտահոսի)
mu = X_train.mean()
sd = X_train.std()
X_train_std = (X_train - mu) / sd
X_test_std = (X_test - mu) / sd

# Կորելացիայի մատրից
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5)
plt.title("Հատկանիշների կորելացիոն մատրից")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

# Bias (ազատ անդամ w0) - ավելացնում ենք 1-երի սյունակ
Xtr = np.c_[np.ones(len(X_train_std)), X_train_std.to_numpy()]
Xte = np.c_[np.ones(len(X_test_std)), X_test_std.to_numpy()]
ytr = y_train.to_numpy()
yte = y_test.to_numpy()

# ---------------------------------------------------------------------
# 3. ՄԵՔԵՆԱՅԱԿԱՆ ԻՐԱԿԱՆԱՑՈՒՄ
# ---------------------------------------------------------------------
def metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return mae, mse, rmse, r2


# 3.1 Մեթոդ 1: Նորմալ հավասարում (Normal Equation, OLS)
#     w = (X^T X)^(-1) X^T y
w_ols = np.linalg.solve(Xtr.T @ Xtr, Xtr.T @ ytr)

# 3.2 Մեթոդ 2: Գրադիենտային անկում (Gradient Descent)
w_gd = np.zeros(Xtr.shape[1])
lr = 0.1
epochs = 3000
N = len(Xtr)
loss_history = []

for epoch in range(epochs):
    pred = Xtr @ w_gd
    error = pred - ytr
    grad = (Xtr.T @ error) / N
    w_gd -= lr * grad
    loss_history.append(np.mean(error ** 2))
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} -> Train MSE: {loss_history[-1]:.4f}")

# ---------------------------------------------------------------------
# 4. ԳՆԱՀԱՏՈՒՄ
# ---------------------------------------------------------------------
results = []
for name, w in [("Normal Equation (OLS)", w_ols), ("Gradient Descent", w_gd)]:
    tr = metrics(ytr, Xtr @ w)
    te = metrics(yte, Xte @ w)
    results.append([name, *tr, *te])

res_df = pd.DataFrame(
    results,
    columns=["Մոդել", "Train MAE", "Train MSE", "Train RMSE", "Train R2",
             "Test MAE", "Test MSE", "Test RMSE", "Test R2"],
)
print("\n--- ԱՐԴՅՈՒՆՔՆԵՐԻ ՀԱՄԵՄԱՏՈՒԹՅՈՒՆ ---")
print(res_df.round(4).to_string(index=False))

# Ստուգում sklearn-ի հետ (պետք է համընկնի OLS-ի հետ)
sk = LinearRegression().fit(X_train_std, y_train)
sk_r2 = metrics(yte, sk.predict(X_test_std))[3]
print(f"\nsklearn LinearRegression Test R2: {sk_r2:.4f} (ստուգման համար)")

# Հիմնական մոդել՝ OLS
w = w_ols
y_pred = Xte @ w

# ---------------------------------------------------------------------
# 5. ԱՐԴՅՈՒՆՔՆԵՐԻ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ
# ---------------------------------------------------------------------
# 5.1 Իրական և կանխատեսված արժեքներ
plt.figure(figsize=(8, 6))
plt.scatter(yte, y_pred, alpha=0.3, color="blue", label="Կանխատեսումներ")
lims = [min(yte.min(), y_pred.min()), max(yte.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=2, label="Իդեալական գիծ (y = x)")
plt.xlabel("Իրական արժեքներ")
plt.ylabel("Կանխատեսված արժեքներ")
plt.title("Իրական և կանխատեսված գների համեմատություն")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# 5.2 Սխալների (residuals) բաշխում
residuals = yte - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=50, color="purple")
plt.xlabel("Սխալ (Իրական - Կանխատեսված)")
plt.ylabel("Հաճախականություն")
plt.title("Մոդելի սխալների բաշխումը")
plt.grid(True)
plt.savefig("residuals_distribution.png", dpi=150)
plt.show()

# 5.3 Ուսուցման կորուստի կոր (Gradient Descent)
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Train MSE")
plt.title("Գրադիենտային անկման կորուստը")
plt.grid(True)
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# 5.4 Հատկանիշների կարևորություն (ստանդարտացված գործակիցներ)
importance = pd.Series(w[1:], index=X.columns)
importance = importance.reindex(importance.abs().sort_values().index)

plt.figure(figsize=(8, 5))
colors = ["green" if v > 0 else "red" for v in importance.values]
importance.plot(kind="barh", color=colors)
plt.xlabel("Ստանդարտացված գործակից")
plt.title("Հատկանիշների կարևորություն (կանաչ՝ գինը մեծացնում է, կարմիր՝ փոքրացնում)")
plt.grid(True, axis="x")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nՀատկանիշների կարևորություն (ըստ |գործակից|-ի՝ նվազման կարգով):")
print(importance.reindex(importance.abs().sort_values(ascending=False).index).round(4))

# ---------------------------------------------------------------------
# 6. ԵԶՐԱԿԱՑՈՒԹՅՈՒՆ
# ---------------------------------------------------------------------
mae_t, mse_t, rmse_t, r2_t = metrics(yte, y_pred)
print("\n================ ԵԶՐԱԿԱՑՈՒԹՅՈՒՆ ================")
print(f"Test MAE : {mae_t:.4f}  (միջին բացարձակ սխալ, ~{mae_t * 100:.0f} հազար $)")
print(f"Test RMSE: {rmse_t:.4f}")
print(f"Test R2  : {r2_t:.4f}  (մոդելը բացատրում է գների տատանման ~{r2_t * 100:.0f}%-ը)")
print("Գծային մոդելը չի գրավում գների և հատկանիշների ոչ գծային կապերը, ")
print("իսկ dataset-ում գները կտրված են 5.0-ի վրա, ինչը սահմանափակում է ճշտությունը։")
