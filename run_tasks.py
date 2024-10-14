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

for i in range(len(task_list)):
    cur_task = task_list[i]

    print(f"{task_names[i]}")
    print("======================================")
    # (seed, lam, angles, targets, targetp, layers, options, sim_dtype, geo_dtype, device
    print("Neural Network")
    print("--------------------------------------")
    t = time.time()
    tNN_des, tNN_cost, tNN_des_hist = NN_optim_pol(cur_task.seed, cur_task.lam, cur_task.angles, 
                                                   cur_task.targets, cur_task.targetp, 
                                                   cur_task.layers, cur_task.options, 
                                                   cur_task.sim_dtype, cur_task.geo_dtype, device)
    print(f"NN optimisation finished in {time.time() - t:.5f} s")
    print("======================================")
    print("Pixel")
    print("--------------------------------------")
    t = time.time()
    tpx_des, tpx_cost, tpx_des_hist = pixel_optim_pol(cur_task.seed, cur_task.lam, cur_task.angles, 
                                                      cur_task.targets, cur_task.targetp, 
                                                      cur_task.layers, cur_task.options, 
                                                      cur_task.sim_dtype, cur_task.geo_dtype, device)
    print(f"Pixel optimisation finished in {time.time() - t:.5f} s")
    print("======================================")

    compare_cost(tNN_cost, tpx_cost, f"{RES_FOLD}{task_names[i]}_cost_compare.png")
    compare_final_designs(tNN_des, tpx_des, f"{RES_FOLD}{task_names[i]}_design_compare.png")