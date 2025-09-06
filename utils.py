import torch


def get_optimizer_scheduler(optimizer_name, scheduler_name):
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD

    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")
    return optimizer, scheduler
