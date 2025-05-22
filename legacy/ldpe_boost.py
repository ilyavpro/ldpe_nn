import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 0. Создание папки для логов
logs_root = r"C:\Users\Ilya\Desktop\AES\code\ldpe_nn\logs_xgb"
run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(logs_root, run_time)
os.makedirs(log_dir, exist_ok=True)
print(f"Логи и результаты будут сохранены в: {log_dir}")

# 1. Загрузка и подготовка данных
data = pd.read_csv(r'C:\Users\Ilya\Desktop\AES\code\ldpe_nn\datasets\сombined_df_grade_1_6_21_05.csv')

X_cols = [
    'E2FD_TEMP','E2FD_FLOW','PFR1_TEMP','INIFD1_FLOW1','INIFD1_FLOW2',
    'MWN_LDPE','MWW_LDPE','FSCBN_LDPE','TMAX_PFR1','TEMP_OUT','FLOW_LDPE'
]
y_cols = [
    'FRPRE_EXP_0','FRPRE_EXP_1','FRPRE_EXP_2','FRPRE_EXP_3','FRPRE_EXP_4',
    'FRPRE_EXP_5','FRPRE_EXP_6','FRPRE_EXP_7','FRPRE_EXP_8','FRPRE_EXP_9'
]

X = data[X_cols].values
y = data[y_cols].values

# 2. Масштабирование
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)
X_scaled = X_scaler.transform(X)
y_scaled = y_scaler.transform(y)

# 3. Train/Test split (валидацию пропускаем для простоты)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 4. Обучение моделей по каждому выходу
models = []
test_preds = []

for i, col in enumerate(y_cols):
    print(f"=== Обучаем модель для {col} ===")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.7,
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_train, y_train[:, i])
    models.append(model)
    test_pred = model.predict(X_test)
    test_preds.append(test_pred)

# Собираем обратно матрицу предсказаний
test_preds = np.column_stack(test_preds)

# 5. Инвертируем стандартизацию (чтобы метрики были как в исходном масштабе)
y_test_orig = y_scaler.inverse_transform(y_test)
test_preds_orig = y_scaler.inverse_transform(test_preds)

np.save(os.path.join(log_dir, "y_test_orig.npy"), y_test_orig)
np.save(os.path.join(log_dir, "y_pred_orig.npy"), test_preds_orig)

# 6. Логирование метрик
print("\n=== Тестовые метрики по каждому выходу (XGBoost) ===")
results = []
for i, col in enumerate(y_cols):
    true = y_test_orig[:, i]
    pred = test_preds_orig[:, i]
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    print(f"{col:>12}: RMSE={rmse:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")
    results.append({"output": col, "RMSE": rmse, "MAE": mae, "MSE": mse, "R2": r2})

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(log_dir, "test_metrics.csv"), index=False)
