import os
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from dataset import MuSTC
from torch.nn import Module
from torch.optim import Optimizer


class Trainer:
  def __init__(
      self,
      model: Module,
      teacher: Module,
      criterion: Module,
      optimizer: Optimizer,
      *dataset_config,
      scheduler=None,
      patience=None,
      details=None
  ):
    self.model = model
    self.teacher = teacher
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.patience = patience
    self.details = details
    self.__dataset = lambda mode: MuSTC(
        'must-c',
        *dataset_config,
        mode == 'eval'
    )
    is_cuda = next(self.model.parameters()).is_cuda
    self.device = torch.device('cuda' if is_cuda else 'cpu')

  def state_dict(self):
    keys = (
        'optimizer',
        *(('scheduler',) if self.scheduler else ())
    )
    state_dict = {k: getattr(self, k).state_dict() for k in keys}
    if hasattr(self, 'running_loss'):
      state_dict['loss'] = self.running_loss
    return state_dict

  def save(self, *details, state_dict=None):
    os.makedirs('./weights', exist_ok=True)
    fname = f"{'_'.join((f'./weights/{self.model.name}_{self.details}', *details))}.pt"
    state_dict = state_dict or self.state_dict()
    torch.save(state_dict, fname)

  def load(self, *details):
    fname = f"{'_'.join((f'./weights/{self.model.name}', *details))}.pt"
    weights = torch.load(fname)
    for module in ('model', 'optimizer'):
      getattr(self, module).load_state_dict(weights[module])

  def train(self, num_epochs=32, batch_size=32):
    modes = 'train', 'dev'
    datasets = (self.__dataset(mode) for mode in modes)
    loaders = (DataLoader(d, batch_size, True) for d in datasets)
    loaders = {k: v for k, v in zip(modes, loaders)}

    self.running_loss = {mode: [] for mode in modes}
    best = [None, 1e14]
    details = f'e={num_epochs}'
    impatience = 0

    try:
      for e in range(num_epochs):
        for mode in modes:
          l = 0

          swa, swa_running = self.scheduler.get('SWA') \
              if self.scheduler else (None for _ in range(2))
          model = swa.averaged_model if mode == 'eval'\
              and swa and swa_running else self.model

          loader = loaders[mode]
          with torch.set_grad_enabled(mode == 'train'):
            for x, y in loader:
              self.optimizer.zero_grad()
              x, y = (i.to(self.device) for i in (x, y))
              pred, encoder_out, _ = model(x)
              _, text_encoder_out, _ = self.teacher(x)
              loss = self.criterion(pred, y, x, encoder_out, text_encoder_out)
              if mode == 'train':
                loss.backward()
                self.optimizer.step()
              l += loss.item()
          print(f'Average loss: {l/len(loader)} on epoch {e+1} ({mode}).')
          self.running_loss[mode].append(l / len(loader))

          if mode == 'train' and self.scheduler is not None:
            self.scheduler.step()
            if swa and swa_running:
              swa.update_bn(loaders['train'], self.device)
          elif l < best[1]:
            impatience = 0
            best = deepcopy(self.state_dict()), l
            details = f'e={e+1}'
          else:
            impatience += 1
            if impatience == self.patience:
              raise StopIteration
          print(flush=True)
    except StopIteration:
      print('Early stopping triggered!', flush=True)

    self.save(details, state_dict=best[0])
    return self.state_dict()
