import torch
import torch.nn.functional as F
from geomloss import SamplesLoss

config = {}


class OTConfig:
  def __init__(self, loss: str, p: int, blur: float, scaling: float):
    self.loss = loss
    self.p = p
    self.blur = blur
    self.scaling = scaling


class CTCWassersteinLoss:
  def __init__(
      self,
      ctc_weight: float,
      ot_weight: float,
      ot_weight_text: float,
      ot_weight_speech: float,
      ot_sampling_config: OTConfig,
      ot_weight_embed: float,
      ot_position_weight: float,
      attn_weight_text: float,
      attn_weight_speech: float,
      gamma: float,  # label smoothing
      pad_idx: int,
      eos_idx: int
  ):
    super().__init__()
    self.ctc_weight = ctc_weight
    self.ot_weight = ot_weight
    self.ot_weight_text = ot_weight_text
    self.ot_weight_speech = ot_weight_speech
    self.ot_sampling_config = vars(ot_sampling_config)
    self.ot_weight_embed = ot_weight_embed
    self.attn_weight_text = attn_weight_text
    self.attn_weight_speech = attn_weight_speech
    self.gamma = gamma
    self.pad_idx = pad_idx
    self.eos_idx = eos_idx

    if ot_position_weight > 0.0:
      self.ot_position_weight = torch.nn.Parameter(
          torch.tensor([ot_position_weight]))

    if self.ot_weight > 0.0 or self.ot_weight_text > 0.0 or self.ot_weight_speech:
      self.ot_loss = SamplesLoss(**self.ot_sampling_config)
    if self.ot_weight_embed > 0.0:
      self.ot_loss_embed = SamplesLoss(**self.ot_sampling_config)

  def forward(self, model, sample, reduce=True):
    net_input = sample["net_input"]
    text_mode = False if net_input['src_tokens'] is not None and net_input.get(
        'src_txt_tokens') is not None else True
    masked_tokens = None
    if getattr(sample, "masked_target", None) is not None:
      masked_tokens = sample["masked_target"].ne(self.pad_idx)

    net_output, encoder_out = model(
        **net_input,
        masked_tokens=masked_tokens,
        use_encoder_outputs=True
    )

    loss = 0.0

    if self.attn_weight_text > 0.0:
      ce_loss_text = self.ce(model, net_output, sample, reduce=reduce, idx=2)
      loss += self.attn_weight_text * ce_loss_text

    if not text_mode:
      if self.attn_weight_speech > 0.0:
        ce_loss_speech = self.ce(
            model,
            net_output,
            sample,
            reduce=reduce,
            idx=0,
        )
        loss += self.attn_weight_speech * ce_loss_speech

      if self.ctc_weight > 0.0:
        ctc_loss = self.ctc(model, net_output, encoder_out, net_input)
        loss += self.ctc_weight * ctc_loss

      if self.ot_weight > 0.0:
        wass_loss = self.ot(self.ot_loss, encoder_out)
        loss += self.ot_weight * wass_loss

      if self.ot_weight_embed > 0.0:
        wass_loss_embed = self.ot(
            self.ot_loss_embed,
            encoder_out,
        )
        loss += self.ot_weight_embed * wass_loss_embed

      if self.ot_weight_text > 0.0 or self.ot_weight_speech > 0.0:
        speech_out = None
        if isinstance(encoder_out, tuple):
          speech_out = encoder_out[0]["encoder_out"][0]
        else:
          speech_out = encoder_out["encoder_out"][0]
        if self.ot_weight_speech > 0.0:
          wass_loss_st = self.ot_loss(
              speech_out.float().transpose(0, 1).contiguous(),
              net_output[1]["before_out_proj"].transpose(
                  0, 1).contiguous()
          ).sum()
          loss += wass_loss_st * self.ot_weight_speech

        if self.ot_weight_text > 0.0 and model.num_updates > 5000:
          speech_decoder_out = net_output[0][0]
          text_decoder_out = net_output[0][-1].detach()
          wass_loss_text = self.ot_loss(speech_decoder_out.contiguous(),
                                        text_decoder_out.contiguous()).sum()
          loss += wass_loss_text * self.ot_weight_text

  def lprobs(self, model, net_output, sample, idx):
    lprobs = model.get_normalized_probs(
        net_output, log_probs=True, idx=idx)
    target = model.get_targets(sample, net_output)
    return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

  def compute_accuracy(self, lprobs, target):
    mask = target.ne(self.padding_idx)
    n_correct = torch.sum(
        lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
    )
    total = torch.sum(mask)
    return n_correct, total

  def ce(self, model, net_output, sample, reduce, idx):
    lprobs, target = self.lprobs(model, net_output, sample, idx=idx)
    return F.nll_loss(
        lprobs,
        target,
        ignore_index=self.pad_idx,
        reduce=reduce,
    )

  def ctc(self, model, net_output, encoder_out, net_input):
    lprobs = model.get_normalized_probs(
        net_output, log_probs=True, idx=1,
    ).contiguous()

    speech_encoder_out = encoder_out[0] if isinstance(
        encoder_out, tuple) else encoder_out
    if speech_encoder_out["encoder_padding_mask"]:
      non_padding_mask = ~speech_encoder_out["encoder_padding_mask"][0]
      input_lengths = non_padding_mask.long().sum(-1)
    else:
      input_lengths = lprobs.new_full(
          (lprobs.size(1),), lprobs.size(0), dtype=torch.long
      )
    pad_mask = net_input["src_txt_tokens"] not in (self.pad_idx, self.eos_idx)
    targets_flat = net_input["src_txt_tokens"].masked_select(pad_mask)
    target_lengths = pad_mask.sum(-1)

    with torch.backends.cudnn.flags(enabled=False):
      ctc_loss = F.ctc_loss(
          lprobs,
          targets_flat,
          input_lengths,
          target_lengths,
          reduction="sum",
          zero_infinity=True,
      )
    kldiv_loss = 0.0
    if self.gamma > 0:
      kldiv_loss = F.kl_div(
          lprobs.transpose(0, 1),
          torch.full_like(lprobs.transpose(0, 1),
                          1 / (lprobs.size(-1) - 1)),
          reduction="batchmean",
      )
    return (1. - self.gamma) * ctc_loss + self.gamma * kldiv_loss

  def ot(self, ot_loss, encoder_out):
    speech_out = encoder_out[0]["encoder_out"][0]
    text_out = encoder_out[-1]["encoder_out"][0]

    speech_out, text_out = (out / torch.linalg.norm(
        out, dim=- 1, keepdim=True
    ) for out in (speech_out, text_out))
    text_out = text_out / \
        torch.linalg.norm(text_out, dim=-1, keepdim=True)
    if self.ot_position_weight[0] > 0.0:
      S, B, _ = speech_out.size()
      T = text_out.size()[0]
      speech_lens = encoder_out[0]["input_lengths"][0]
      text_lens = encoder_out[1]["src_lengths"][0].squeeze(-1)
      speech_lens[speech_lens <= 1] = 2
      text_lens[text_lens <= 1] = 2
      speech_pos, text_pos = (torch.matmul(
          torch.tensor(
              range(dim),
              dtype=torch.float,
              device=speech_out.device
          ).unsqueeze(-1),
          torch.ones((1, B), device=speech_out.device)
      ) for dim in (S, T))
      speech_pos, text_pos = (
          pos / (lens - 1).unsqueeze(0)
          for pos, lens in zip(
              (speech_pos, text_pos), (speech_lens, text_lens)))
      speech_pos[speech_pos > 1] = 1e9
      text_pos[text_pos > 1] = 1e9
      speech_out, text_out = (torch.cat(
          (out, pos.unsqueeze(-1)), dim=-1)
          for out, pos in zip((speech_out, text_out), (speech_pos, text_pos)))
    with torch.cuda.amp.autocast(enabled=False):
      wass_loss = ot_loss(
          speech_out.float().transpose(0, 1).contiguous(),
          text_out.float().transpose(0, 1).contiguous()
      ).sum()
    return wass_loss
