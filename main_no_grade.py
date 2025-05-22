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
VARIATION = 0.2  # только для input_params (реакционные параметры)
batch_size = 50

# --- Варьируемые fixed-параметры с индивидуальным диапазоном для каждого
fixed_input_params = [
    {"name": "E2FD_TEMP",    "path": r"\Data\Streams\E2FD\Input\TEMP\MIXED",        "min": 100,    "max": 120},
    {"name": "E2FD_FLOW",    "path": r"\Data\Streams\E2FD\Input\FLOW\MIXED\E2",     "min": 55000, "max": 65000},
    {"name": "PFR1_TEMP",    "path": r"\Data\Blocks\PFR1\Input\TEMP",               "min": 170,   "max": 165},
    {"name": "INIFD1_FLOW1", "path": r"\Data\Streams\INIFD1\Input\FLOW\MIXED\INI1", "min": 6,     "max": 8},
    {"name": "INIFD1_FLOW2", "path": r"\Data\Streams\INIFD1\Input\FLOW\MIXED\INI2", "min": 1.5,     "max": 2},
]

# --- Варьируемые реакционные параметры
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
file_name = fr"{base_path}\datasets\paramsweep_{datetime.now().strftime('%m-%d_%H-%M')}_test.csv"

# --- Порядок колонок: сначала варьируемые, потом fixed, потом выходы
column_names = (
    [param["name"] for param in input_params] +
    [param["name"] for param in fixed_input_params] +
    [param["name"] for param in output_params]
)
write_header = True

def get_new_sim():
    sim = win32.Dispatch("Apwn.Document")
    sim.InitFromArchive2(model_path)
    return sim

sim = get_new_sim()
batch_time = 0.0

for i in tqdm(range(N_SAMPLES)):
    if i != 0 and i % batch_size == 0:
        sim.Quit()
        sim = get_new_sim()

    row = {}

    # --- Варьируемые параметры (input_params)
    for param in input_params:
        value = np.random.uniform(param["nominal"] * (1 - VARIATION), param["nominal"] * (1 + VARIATION))
        sim.Tree.FindNode(param["path"]).Value = value
        row[param["name"]] = value

    # --- Fixed-параметры в индивидуальных диапазонах
    for param in fixed_input_params:
        value = np.random.uniform(param["min"], param["max"])
        sim.Tree.FindNode(param["path"]).Value = value
        row[param["name"]] = value

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

    curr_batch = min(batch_size, (i % batch_size) + 1)
    start_idx = i + 2 - curr_batch
    if (i + 1) % batch_size == 0 or (i == N_SAMPLES - 1):
        print(f"Steps {start_idx}-{i+1} ({curr_batch} шт): суммарно {batch_time:.2f} сек, среднее {batch_time/curr_batch:.2f} сек/шаг")
        batch_time = 0.0

sim.Quit()
