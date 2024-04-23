import torch
from sacrebleu import BLEU, corpus_bleu as _corpus_bleu


class Metrics:
  @staticmethod
  def average_lagging(delays, src_lens, tgt_lens):
    lag_mask = delays >= src_lens
    lag_mask = torch.nn.functional.pad(lag_mask.T, (1, 0)).T[:-1, :]
    gamma = tgt_lens / src_lens
    lagging = (
        delays - torch.arange(delays.size(0))
        .unsqueeze(1)
        .type_as(delays)
        .expand_as(delays)
        / gamma
    )
    lagging.masked_fill_(lag_mask, 0)
    tau = (1 - lag_mask.type_as(lagging)).sum(dim=0, keepdim=True)
    return lagging.sum(dim=0, keepdim=True) / tau

  @staticmethod
  def corpus_bleu(pred, target):
    bleu = _corpus_bleu(pred, target, tokenize="none")
    return bleu.score

  @staticmethod
  def sentence_bleu(pred, target):
    bleu = _corpus_bleu(pred, target)
    for i in range(1, 4):
      bleu.counts[i] += 1
      bleu.totals[i] += 1
    bleu = BLEU.compute_bleu(
        bleu.counts,
        bleu.totals,
        bleu.sys_len,
        bleu.ref_len,
        smooth_method="exp"
    )
    return bleu.score
