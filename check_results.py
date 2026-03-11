import json
import os

import matplotlib.pyplot as plt
import pandas as pd

files = os.listdir("./train_runs/")

all_data = []

for f_name in files:
    with open(f"./train_runs/{f_name}", "r") as file:
        info = json.load(file)

        info["last_loss"] = info["losses"][-1]

        all_data.append(info)

df = pd.DataFrame(all_data)
df = df.loc[df.groupby("loss_function")["last_loss"].idxmin()]

best_confs = df[["loss_function", "base_lr", "losses", "scheduler"]]

plt.figure(figsize=(14, 4))
for i in range(len(df)):
    row = df.iloc[i]

    print(f"Best config for {row['loss_function'].upper()} -> {row['last_loss']:.5f}")
    print(f"Base learning rate: {row['base_lr']}")
    print(f"Scheduler: ")
    print(f"    Name: {row['scheduler']['scheduler_name']}")
    print(f"    Config: {row['scheduler']['scheduler_conf']}")
    print()

    plt.subplot(1, len(df), i + 1)
    plt.plot(row["losses"])

plt.tight_layout()
plt.show(block=True)
