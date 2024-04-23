import torch
from transformers.modeling_outputs import ModelOutput
from torch.nn import Module
from abc import ABC, abstractmethod


class SequenceToSequenceModel(Module, ABC):
  @property
  @abstractmethod
  def name(self):
    pass

  def __init__(self, processor, model, device: torch.device, model_path: str):
    super(SequenceToSequenceModel, self).__init__()
    self.processor = processor
    self.model = model
    self.device = device
    self.load(model_path)

  def load(self, filepath=None):
    p, m = ('/'.join((filepath or '', module))
            for module in ('processor', 'model'))
    p, m = (path if filepath else self.name for path in (p, m))
    pretrained_cache = 'pretrained_models'
    self.processor = self.processor.from_pretrained(
        p,
        cache_dir=p if filepath else pretrained_cache
    )
    self.model = self.model.from_pretrained(
        m,
        cache_dir=m if filepath else pretrained_cache,
        use_safetensors=True,
        output_hidden_states=True,
        return_dict_in_generate=True
    ).to(self.device)

  def save(self, path):
    for module in ('processor', 'model'):
      self[module].save_pretrained('/'.join(path, module))

  def process_model_output(self, output: ModelOutput):
    decoded_output = self.processor.decode(
        output.sequences[0],
        skip_special_tokens=True
    )
    encoder_out, decoder_out = (output[f'{module}_hidden_states'][-1]
                                for module in ('encoder', 'decoder'))
    return decoded_output, encoder_out, decoder_out
