import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from .sequence_to_sequence import SequenceToSequenceModel


class TextToTextModel(SequenceToSequenceModel):
  name = 'kazandaev/m2m100_418M-finetuned-en-ru'

  def __init__(
      self,
      device: torch.device,
      model_path: str = None
  ):
    super(TextToTextModel, self).__init__(
        M2M100Tokenizer,
        M2M100ForConditionalGeneration,
        device,
        model_path
    )
    self.processor.src_lang = 'en'

  def forward(self, x):
    input_ids = self.processor.encode(x, return_tensors='pt')
    output = self.model.generate(
        input_ids.to(self.device),
        forced_bos_token_id=self.processor.get_lang_id('ru')
    )
    return self.process_model_output(output)
