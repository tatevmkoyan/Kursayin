import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# ==========================================
# 1. ՏՎՅԱԼՆԵՐԻ ՀԱՎԱՔԱԾՈՒ ԵՎ ՆԱԽԱՊԱՏՐԱՍՏՈՒՄ
# ==========================================

# Բեռնում ենք California Housing dataset-ը անմիջապես sklearn-ից
california = fetch_california_housing(as_frame=True)
df = california.frame

# Առանձնացնում ենք X (նկարագրող հատկանիշներ) և y (նպատակային փոփոխական՝ MedHouseVal)
y = df['MedHouseVal'].tolist()
feature_names = [c for c in df.columns if c != 'MedHouseVal']
X = df[feature_names].values.tolist()

# Տվյալների բաժանում Train (80%) և Test (20%)
indices = list(range(len(X)))
random.seed(42)
random.shuffle(indices)

train_size = int(len(indices) * 0.8)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = [X[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
X_test = [X[i] for i in test_idx]
y_test = [y[i] for i in test_idx]

# Տվյալների ստանդարտացում (Standardization)
means = []
stds = []
for col in range(len(X_train[0])):
  column_vals = [row[col] for row in X_train]
  m = sum(column_vals) / len(column_vals)
  s = math.sqrt(sum((v - m) ** 2 for v in column_vals) / len(column_vals))
  means.append(m)
  stds.append(s if s != 0 else 1)


def standardize(row):
  return [(row[i] - means[i]) / stds[i] for i in range(len(row))]


X_train_std = [standardize(r) for r in X_train]
X_test_std = [standardize(r) for r in X_test]


# Bias (ազատ անդամի՝ w0-ի համար 1-երի սյուն) ավելացում
def add_bias(X):
  return [[1] + row for row in X]


X_train_b = add_bias(X_train_std)
X_test_b = add_bias(X_test_std)


# ==========================================
# 2. ՄԱԹԵՄԱՏԻԿԱԿԱՆ ԵՎ ՄԱՏՐԻՑԱՅԻՆ ԳՈՐԾՈՂՈՒԹՅՈՒՆՆԵՐ
# ==========================================
def transpose(M):
  return list(map(list, zip(*M)))


def matmul(A, B):
  result = []
  for i in range(len(A)):
    row = []
    for j in range(len(B[0])):
      s = sum(A[i][k] * B[k][j] for k in range(len(B)))
      row.append(s)
    result.append(row)
  return result


def inverse(M):
  n = len(M)
  aug = [M[i] + [1 if i == j else 0 for j in range(n)] for i in range(n)]
  for i in range(n):
    diag = aug[i][i]
    if diag == 0:
      for k in range(i + 1, n):
        if aug[k][i] != 0:
          aug[i], aug[k] = aug[k], aug[i]
          diag = aug[i][i]
          break
    for j in range(2 * n):
      aug[i][j] /= diag
    for k in range(n):
      if k != i:
        factor = aug[k][i]
        for j in range(2 * n):
          aug[k][j] -= factor * aug[i][j]
  return [row[n:] for row in aug]


# ==========================================
# 3. ԳԾԱՅԻՆ ՌԵԳՐԵՍԻԱՅԻ (OLS) ՈՒՍՈՒՑՈՒՄ
# ==========================================
def train_linear_regression(X, y):
  X_T = transpose(X)
  XTX = matmul(X_T, X)
  XTy = matmul(X_T, [[v] for v in y])
  XTX_inv = inverse(XTX)
  W = matmul(XTX_inv, XTy)
  return [w[0] for w in W]


w = train_linear_regression(X_train_b, y_train)


def predict_row(w, row):
  return sum(w[i] * row[i] for i in range(len(w)))


def predict(w, X):
  return [predict_row(w, row) for row in X]


y_pred = predict(w, X_test_b)


# ==========================================
# 4. ՄՈԴԵԼԻ ՈՐԱԿԻ ԳՆԱՀԱՏՈՒՄ
# ==========================================
def mae(y_true, y_pred):
  return (
      sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true)))
      / len(y_true)
  )


def mse(y_true, y_pred):
  return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(
      y_true
  )


def rmse(y_true, y_pred):
  return math.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
  mean_y = sum(y_true) / len(y_true)
  ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
  ss_tot = sum((v - mean_y) ** 2 for v in y_true)
  return 1 - (ss_res / ss_tot)


print(f'MAE:  {mae(y_test, y_pred):.2f}')
print(f'MSE:  {mse(y_test, y_pred):.2f}')
print(f'RMSE: {rmse(y_test, y_pred):.2f}')
print(f'R²:   {r2(y_test, y_pred):.4f}')


# ==========================================
# 5. ՎԻԶՈՒԱԼԻԶԱՑԻԱ (ԳՐԱՖԻԿՆԵՐ)
# ==========================================

# 1. Իրական vs Կանխատեսված գներ
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red',
    linewidth=2,
    linestyle='--',
)
plt.xlabel('Իրական գներ (Actual Values)')
plt.ylabel('Կանխատեսված գներ (Predicted Values)')
plt.title('Իրական և Կանխատեսված գների համեմատություն')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()

# 2. Սխալների (Residuals) բաշխման հիստոգրամ
residuals = [y_test[i] - y_pred[i] for i in range(len(y_test))]
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple', bins=50)
plt.xlabel('Սխալի չափը (Residual = Actual - Predicted)')
plt.ylabel('Հաճախականություն')
plt.title('Մոդելի սխալների (Residuals) բաշխումը')
plt.grid(True)
plt.savefig('residuals_distribution.png')
plt.show()

# 3. Կորելացիոն մատրից (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Հատկանիշների կորելացիոն մատրից (Heatmap)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()