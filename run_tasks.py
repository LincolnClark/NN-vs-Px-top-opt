import torch
import numpy as np
import time
import os
import importlib
from NN_reparam.NN_optim_pol_dependent import NN_optim_pol
from Pixel_param.pixel_optim_pol_dependent import pixel_optim_pol
from utils.plot_compare import *

TASK_FOLDER = "Tasks"
RES_FOLD = "./Results/"

task_list = []
task_names = []
for task in os.listdir(f"./{TASK_FOLDER}/"):
    if task[0] != "_":
        task_str = task[:-3]
        task_names.append(task_str)
        task_list.append(importlib.import_module(f"{TASK_FOLDER}.{task_str}"))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"There are {len(task_list)} tasks to run")

for i in range(len(task_list)):
    cur_task = task_list[i]

    print(f"{task_names[i]}")
    print("======================================")
    # (seed, lam, angles, targets, targetp, layers, options, sim_dtype, geo_dtype, device
    print("Neural network started")
    t = time.time()
    NN_des, NN_cost, NN_des_hist, NN_ts, NN_tp  = NN_optim_pol(cur_task.seed, cur_task.lam, cur_task.angles, 
                                                               cur_task.targets, cur_task.targetp, 
                                                               cur_task.layers, cur_task.options, 
                                                               cur_task.sim_dtype, cur_task.geo_dtype, device)
    print(f"NN optimisation finished in {time.time() - t:.5f} s")
    print("======================================")
    print("Pixel optimisation started")
    t = time.time()
    px_des, px_cost, px_des_hist, px_ts, px_tp = pixel_optim_pol(cur_task.seed, cur_task.lam, cur_task.angles, 
                                                                 cur_task.targets, cur_task.targetp, 
                                                                 cur_task.layers, cur_task.options, 
                                                                 cur_task.sim_dtype, cur_task.geo_dtype, device)
    print(f"Pixel optimisation finished in {time.time() - t:.5f} s")
    print("======================================")

    compare_cost(NN_cost, px_cost, f"{RES_FOLD}{task_names[i]}_cost_compare.png")
    compare_final_designs(NN_des, px_des, f"{RES_FOLD}{task_names[i]}_design_compare.png")
    compare_performances(NN_ts, NN_tp, px_ts, px_tp, cur_task.targets, cur_task.targetp,
                         cur_task.angles, f"{RES_FOLD}{task_names[i]}_performance_compare.png")
    animate_history(NN_des_hist, f"{RES_FOLD}{task_names[i]}_kappa_NN_ani.gif")
    animate_history(px_des_hist, f"{RES_FOLD}{task_names[i]}_kappa_px_ani.gif")