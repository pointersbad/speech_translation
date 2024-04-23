import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler


class Sequential:
  def __init__(
      self,
      schedulers: list[LRScheduler],
      milestones: tuple[int] = (),
      verbose=False
  ):
    self.schedulers, self.milestones = (
        x if isinstance(x, (list, tuple))
        else (x,) for x in (schedulers, milestones)
    )
    self.verbose = verbose
    self.running_lr = []
    self.current_step = 0
    self.i = 0

  def get(self, name: str):
    scheduler_names = [s.__class__.__name__ for s in self.schedulers]
    si = scheduler_names.index(name) if name in scheduler_names else None
    scheduler = self.schedulers[si] if si else si
    return scheduler, si == self.i

  def step(self):
    self.schedulers[self.i].step()
    if self.verbose:
      lr = self.schedulers[self.i].get_last_lr()
      self.running_lr.append(lr)
      print(f"Applied learning rates: {lr}")
    self.current_step += 1

    guard = self.i < len(self.milestones)
    if guard and self.current_step ==\
            self.milestones[self.i] + sum(self.milestones[:self.i]):
      self.i += 1
      if self.verbose:
        name = self.schedulers[self.i].__class__.__name__
        print(f"Switching the learning rate scheduler to {name}.")
    if self.verbose:
      print()

  def state_dict(self):
    state_dict = {}
    for s in self.schedulers:
      name = s.__class__.__name__
      state_dict[name] = s.state_dict()
    state_dict['lr'] = self.running_lr
    state_dict['step'] = self.current_step
    return state_dict

  def load_state_dict(self, state_dict):
    for scheduler in self.schedulers:
      name = scheduler.__class__.__name__
      scheduler.load_state_dict(state_dict[name])


class SWA:
  def __init__(self, optimizer: Optimizer, lrs: float, model: Module):
    self.scheduler = SWALR(optimizer, lrs)
    self.model = model
    averaged_model = AveragedModel(self.model)
    averaged_model.name = f'averaged_{self.model.name}'
    self.averaged_model = averaged_model

  def step(self):
    self.averaged_model.update_parameters(self.model)
    self.scheduler.step()

  def update_bn(self, loader: DataLoader, device: torch.device):
    update_bn(loader, self.averaged_model, device)

  def state_dict(self):
    state_dict = {
        'model': self.averaged_model.state_dict(),
        'scheduler': self.scheduler.state_dict()
    }
    return state_dict

  def load_state_dict(self, state_dict: dict):
    self.scheduler.load_state_dict(state_dict['scheduler'])

  def __getattr__(self, attr):
    return getattr(self.scheduler, attr)
