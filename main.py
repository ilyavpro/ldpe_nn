import win32com.client as win32
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import time

base_path = r"C:\Users\Ilya\Desktop\AES\code\ldpe_nn"
model_path = fr"{base_path}\model\ldpe.bkp"
N_SAMPLES = 500
VARIATION = 0.2
batch_size = 50

# --- Задать grade для расчета:
grade = 6  # <---- Меняешь только здесь!

# --- Фиксированные параметры
fixed_input_params = [
    {"name": "E2FD_TEMP",    "path": r"\Data\Streams\E2FD\Input\TEMP\MIXED"},
    {"name": "E2FD_FLOW",    "path": r"\Data\Streams\E2FD\Input\FLOW\MIXED\E2"},
    {"name": "PFR1_TEMP",    "path": r"\Data\Blocks\PFR1\Input\TEMP"},
    {"name": "INIFD1_FLOW1", "path": r"\Data\Streams\INIFD1\Input\FLOW\MIXED\INI1"},
    {"name": "INIFD1_FLOW2", "path": r"\Data\Streams\INIFD1\Input\FLOW\MIXED\INI2"},
]

# --- Значения фиксированных параметров для разных grade
grade_values = {
    1: {
        "E2FD_TEMP":    100,
        "E2FD_FLOW":    65000,
        "PFR1_TEMP":    170,
        "INIFD1_FLOW1": 6,
        "INIFD1_FLOW2": 1.5,
    },
    2: {
        "E2FD_TEMP":    120,
        "E2FD_FLOW":    65000,
        "PFR1_TEMP":    170,
        "INIFD1_FLOW1": 6,
        "INIFD1_FLOW2": 1.5,
    },
    3: {
        "E2FD_TEMP":    100,
        "E2FD_FLOW":    55000,
        "PFR1_TEMP":    170,
        "INIFD1_FLOW1": 6,
        "INIFD1_FLOW2": 1.5,
    },
    4: {
        "E2FD_TEMP":    100,
        "E2FD_FLOW":    65000,
        "PFR1_TEMP":    165,
        "INIFD1_FLOW1": 6,
        "INIFD1_FLOW2": 1.5,
    },
    5: {
        "E2FD_TEMP":    100,
        "E2FD_FLOW":    65000,
        "PFR1_TEMP":    170,
        "INIFD1_FLOW1": 8,
        "INIFD1_FLOW2": 1.5,
    },
    6: {
        "E2FD_TEMP":    100,
        "E2FD_FLOW":    65000,
        "PFR1_TEMP":    170,
        "INIFD1_FLOW1": 6,
        "INIFD1_FLOW2": 2,
    },
    # Можно добавить ещё любые grade
}

# --- Варьируемые параметры (добавлен FRPRE_EXP_0)
input_params = [
    {"name": "FRPRE_EXP_0", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#0", "nominal": 3.8607E-06},
    {"name": "FRPRE_EXP_1", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#1", "nominal": 3.7905E-09},
    {"name": "FRPRE_EXP_2", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#2", "nominal": 250000000},
    {"name": "FRPRE_EXP_3", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#3", "nominal": 250000000},
    {"name": "FRPRE_EXP_4", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#4", "nominal": 1250000},
    {"name": "FRPRE_EXP_5", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#5", "nominal": 1240000},
    {"name": "FRPRE_EXP_6", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#6", "nominal": 60700000},
    {"name": "FRPRE_EXP_7", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#7", "nominal": 2500000000},
    {"name": "FRPRE_EXP_8", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#8", "nominal": 2500000000},
    {"name": "FRPRE_EXP_9", "path": r"\Data\Reactions\Reactions\R1\Input\FRPRE_EXP\#9", "nominal": 1300000000},
]

# --- Выходные параметры
output_params = [
    {"name": "MWN_LDPE",   "path": r"\Data\Streams\LPSB\Output\COMP_ATTR\MWN\MWN\LDPE\MIXED"},
    {"name": "MWW_LDPE",   "path": r"\Data\Streams\LPSB\Output\COMP_ATTR\MWW\MWW\LDPE\MIXED"},
    {"name": "FSCBN_LDPE", "path": r"\Data\Streams\LPSB\Output\COMP_ATTR\FSCBN\FSCB\LDPE\MIXED"},
    {"name": "TMAX_PFR1",  "path": r"\Data\Blocks\PFR1\Output\TMAX"},
    {"name": "TEMP_OUT",   "path": r"\Data\Streams\OUT1\Output\TEMP_OUT\MIXED"},
    {"name": "FLOW_LDPE",  "path": r"\Data\Streams\LPSB\Output\MASSFLOW\MIXED\LDPE"},
]

os.makedirs(fr"{base_path}\datasets", exist_ok=True)
file_name = fr"{base_path}\datasets\grade_{grade}_{datetime.now().strftime('%m-%d_%H-%M')}_test.csv"

# --- Колонки с grade в начале
column_names = (
    ['grade'] +
    [param["name"] for param in input_params] +
    [param["name"] for param in fixed_input_params] +
    [param["name"] for param in output_params]
)
write_header = True

def get_new_sim():
    sim = win32.Dispatch("Apwn.Document")
    sim.InitFromArchive2(model_path)
    return sim

# --- значения фиксированных параметров для выбранного grade
fixed_values_dict = grade_values[grade]
fixed_values = [fixed_values_dict[param["name"]] for param in fixed_input_params]

sim = get_new_sim()
batch_time = 0.0

for i in tqdm(range(N_SAMPLES)):
    if i != 0 and i % batch_size == 0:
        sim.Quit()
        sim = get_new_sim()

    row = {}
    row['grade'] = grade

    # --- Варьируемые параметры
    for param in input_params:
        value = np.random.uniform(param["nominal"] * (1 - VARIATION), param["nominal"] * (1 + VARIATION))
        sim.Tree.FindNode(param["path"]).Value = value
        row[param["name"]] = value

    # --- Фиксированные параметры
    for param, val in zip(fixed_input_params, fixed_values):
        sim.Tree.FindNode(param["path"]).Value = val
        row[param["name"]] = val

    t_start = time.time()
    sim.Engine.Run2()
    t_end = time.time()
    elapsed = t_end - t_start
    batch_time += elapsed

    for param in output_params:
        row[param["name"]] = sim.Tree.FindNode(param["path"]).Value

    df_row = pd.DataFrame([row], columns=column_names)
    df_row.to_csv(file_name, mode='a', header=write_header, index=False)
    write_header = False

    # Корректный диапазон для print
    curr_batch = min(batch_size, (i % batch_size) + 1)
    start_idx = i + 2 - curr_batch
    if (i + 1) % batch_size == 0 or (i == N_SAMPLES - 1):
        print(f"Steps {start_idx}-{i+1} ({curr_batch} шт): суммарно {batch_time:.2f} сек, среднее {batch_time/curr_batch:.2f} сек/шаг")
        batch_time = 0.0

sim.Quit()
