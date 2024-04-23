import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from .sequence_to_sequence import SequenceToSequenceModel


class SpeechToTextModel(SequenceToSequenceModel):
  name = 'distil-whisper/distil-large-v3'

  def __init__(
      self,
      device: torch.device,
      sample_rate: int,
      model_path: str = None
  ):
    super(SpeechToTextModel, self).__init__(
        WhisperProcessor,
        WhisperForConditionalGeneration,
        device,
        model_path
    )
    self.sample_rate = sample_rate

  def forward(self, x):
    input_features = self.processor(
        x,
        sampling_rate=self.sample_rate,
        return_tensors='pt'
    ).input_features
    output = self.model.generate(
        input_features.to(self.device),
        language='english',
        task='transcribe',
        prompt_lookup_num_tokens=10,
        use_cache=True,
        do_sample=True,
        temperature=0.4,
        repetition_penalty=1.2
    )
    return self.process_model_output(output)
