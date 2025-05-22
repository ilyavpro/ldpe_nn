import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# 0. Создание папки для логов
logs_root = r"C:\Users\Ilya\Desktop\AES\code\ldpe_nn\logs_nn"
run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(logs_root, run_time)
os.makedirs(log_dir, exist_ok=True)
print(f"Логи и результаты будут сохранены в: {log_dir}")

# 1. Загрузка и подготовка данных
data = pd.read_csv(r'C:\Users\Ilya\Desktop\AES\code\ldpe_nn\datasets\partly_05-22_17-15.csv')

X_cols = [
    'E2FD_TEMP','E2FD_FLOW','PFR1_TEMP','INIFD1_FLOW1','INIFD1_FLOW2',
    'MWN_LDPE','MWW_LDPE','FSCBN_LDPE','TMAX_PFR1','TEMP_OUT','FLOW_LDPE'
]

y_cols = [
    'FRPRE_EXP_2','FRPRE_EXP_4','FRPRE_EXP_7','FRPRE_EXP_8'
]

X = data[X_cols].values
y = data[y_cols].values

# 2. Масштабирование
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)
X_scaled = X_scaler.transform(X)
y_scaled = y_scaler.transform(y)

# 3. Dataset и DataLoader
class LDPE_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

full_dataset = LDPE_Dataset(X_scaled, y_scaled)

n_total = len(full_dataset)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val
train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# 4. Модель
class LDPE_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 4) 
        )
    def forward(self, x):
        return self.net(x)

model = LDPE_Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 5. Обучение с логированием
num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    train_loss = running_loss / len(train_ds)
    train_losses.append(train_loss)
    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            running_val_loss += val_loss.item() * X_val.size(0)
    val_loss = running_val_loss / len(val_ds)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

# 6. Сохранение истории обучения
log_df = pd.DataFrame({
    "epoch": np.arange(1, num_epochs+1),
    "train_loss": train_losses,
    "val_loss": val_losses
})
log_df.to_csv(os.path.join(log_dir, "training_log.csv"), index=False)

# 7. Финальные метрики на тестовой выборке и сохранение предсказаний
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch).cpu().numpy()
        y_pred.append(preds)
        y_true.append(y_batch.cpu().numpy())
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# Inverse transform to original scale
y_true_orig = y_scaler.inverse_transform(y_true)
y_pred_orig = y_scaler.inverse_transform(y_pred)

# Сохраним настоящие и предсказанные значения для анализа/графиков
np.save(os.path.join(log_dir, "y_true_orig.npy"), y_true_orig)
np.save(os.path.join(log_dir, "y_pred_orig.npy"), y_pred_orig)

print("\n=== Тестовые метрики по каждому выходу ===")
for i, col in enumerate(y_cols):
    true = y_true_orig[:, i]
    pred = y_pred_orig[:, i]
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - pred))
    tss = np.sum((true - np.mean(true)) ** 2)
    rss = np.sum((true - pred) ** 2)
    r2 = 1 - rss / tss if tss > 0 else float('nan')
    print(f"{col:>12}: RMSE={rmse:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")
