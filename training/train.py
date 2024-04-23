import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, LambdaLR
from models.speech_to_text import SpeechToTextModel
from models.text_to_text import TextToTextModel
from .criterion import CTCWassersteinLoss, OTConfig
from .trainer import Trainer
from .scheduler import Sequential, SWA
from signal import *


import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {str(device).upper()}.\n")


def linear(optimizer, *_):
  return Sequential(
      schedulers=(LambdaLR(optimizer, lambda _: 1)),
      verbose=True), 'baseline'


def multistep(optimizer, *_):
  warmup_duration = 8
  warmup = LambdaLR(optimizer, lambda e: e / warmup_duration)
  step = MultiStepLR(optimizer, [8, 24, 32], 0.25)
  return Sequential(
      schedulers=(warmup, step),
      milestones=(warmup_duration),
      verbose=True), 'warmup+multistep'


def multistep_swa(optimizer, model, lr):
  warmup_duration = 4
  step_duration = 16
  warmup = LambdaLR(optimizer, lambda e: e / warmup_duration)
  step = MultiStepLR(optimizer, [4, 12], 0.25)
  swa = SWA(optimizer, lr * 2e-2, model)
  return Sequential(
      schedulers=(warmup, step, swa),
      milestones=(warmup_duration, step_duration),
      verbose=True), 'warmup+multistep+swa'


def cosine_annealing_swa(optimizer, model, lr):
  warmup_duration = 4
  annealing_duration = 72
  warmup = LambdaLR(optimizer, lambda e: e / warmup_duration)
  annealing = CosineAnnealingWarmRestarts(optimizer, 12, 2, lr * 1e-2)
  swa = SWA(optimizer, lr * 2e-2, model)
  return Sequential(
      schedulers=(warmup, annealing, swa),
      milestones=(warmup_duration, annealing_duration),
      verbose=True), 'warmup+cosine_annealing_warm_restarts'


for get_scheduler in (
    linear,
    multistep,
    multistep_swa,
    cosine_annealing_swa
):
  for lr, _lr in ((8e-3, 6e-5),):
    epochs = 28
    model = SpeechToTextModel()
    teacher = TextToTextModel()
    optimizer = AdamW(
        [
            {'params': teacher.parameters(), 'lr': _lr},
            {'params': model.parameters(), 'lr': lr}
        ],
        betas=(0.9, 0.98),
        weight_decay=4e-4
    )
    scheduler, details = get_scheduler(optimizer, model, lr)
    details = '_'.join((details, f'lr={lr}'))
    criterion = CTCWassersteinLoss(
        1,
        0.1,
        OTConfig("sinkhorn", 2, 0.05, 0.5),
        model.processor.tokenizer.pad_token_id,
        model.processor.tokenize.eos_token_id,
        ot_position_weight=1,
        attn_weight_speech=0.1,
        attn_weight_text=0.075,
        gamma=0.1
    )
    trainer = Trainer(
        model,
        teacher,
        criterion,
        optimizer,
        scheduler=scheduler,
        patience=8,
        details=details
    )

    for sig in (SIGABRT, SIGILL, SIGINT, SIGSEGV, SIGTERM):
      def clean(*_):
        trainer.save(sig.name)
        os._exit(0)
      signal(sig, clean)

    state_dict = trainer.train(epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning rates')
    for running_lr in np.transpose(state_dict['scheduler']['lr']):
      ax2.plot(running_lr)
    ax1.plot(state_dict['loss']['train'], color='red')
    ax1.plot(state_dict['loss']['eval'], color='green')
    fig.tight_layout()
    os.makedirs('visuals', exist_ok=True)
    fig.savefig(f'visuals/{details}.png')
    fig.clear()
