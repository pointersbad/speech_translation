import numpy as np
from SimulEval.simuleval.utils import entrypoint
from SimulEval.simuleval.agents import SpeechToTextAgent
from SimulEval.simuleval.agents.actions import WriteAction, ReadAction


@entrypoint
class WaitK(SpeechToTextAgent):
  def __init__(
      self,
      model,
      k,
      segment_length,
      continuous
  ):
    super().__init__(k)
    self.model = model
    self.k = k
    self.segment_length = segment_length
    self.continuous = continuous

  def policy(self, states=None):
    if states is None:
      states = self.states
      length_in_seconds = float(len(states.source)) / states.source_sample_rate \
          if states.source_sample_rate != 0 else 0
    if not states.source_finished:
      if length_in_seconds * 1e3 / self.segment_length < self.k:
        return ReadAction()

    audio = np.array(states.source).astype(np.float32)
    prediction = self.model(audio)

    if not states.source_finished and self.continuous > 0:
      prediction = prediction[: self.continuous]

    return WriteAction(
        content=' '.join(prediction),
        finished=states.source_finished,
    )
