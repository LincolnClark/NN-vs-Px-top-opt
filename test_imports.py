import os
import importlib

TASK_FOLDER = "Tasks"

task_list = []
task_names = []
for task in os.listdir(f"./{TASK_FOLDER}/"):
    if task[0] != "_":
        task_str = task[:-3]
        task_names.append(task_str)
        task_list.append(importlib.import_module(f"{TASK_FOLDER}.{task_str}"))

print(task_names)