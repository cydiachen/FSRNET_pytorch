import os
import torch

def save_checkpoint(model, epoch, model_path, iteration, prefix=""):
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    model_out_path = model_path + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

# 后期检查一下是否需要单独写一下

def save_Hyperparameter():
    pass