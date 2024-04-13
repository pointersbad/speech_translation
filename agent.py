from typing import Optional
from SimulEval.simuleval.agents.states import AgentStates
from SimulEval.simuleval.utils import entrypoint
from SimulEval.simuleval.data.segments import SpeechSegment
from SimulEval.simuleval.agents import SpeechToTextAgent
from SimulEval.simuleval.agents.actions import WriteAction, ReadAction

import numpy


@entrypoint
class WaitK(SpeechToTextAgent):
  def __init__(self, k, src_segment_size,
               src_language, continuous, model, task):
    super().__init__(k)
    self.k = k
    self.source_segment_size = src_segment_size
    self.source_language = src_language
    self.continuous_write = continuous
    self.model = model
    self.task = task

  def policy(self, states: Optional[AgentStates] = None):
    if states is None:
      states = self.states

    if states.source_sample_rate == 0:
      length_in_seconds = 0
    else:
      length_in_seconds = float(
          len(states.source)) / states.source_sample_rate

    if not states.source_finished:
      if (
          length_in_seconds * 1000 / self.source_segment_size
      ) < self.k:
        return ReadAction()

    previous_translation = " ".join(states.target)
    options = self.model.DecodingOptions(
        prefix=previous_translation,
        language=self.source_language,
        without_timestamps=True,
        fp16=False,
    )

    # We encode the whole audio to get the full transcription each time a
    # new audio chunk is received.
    audio = self.model.pad_or_trim(
        numpy.array(states.source).astype("float32"))
    mel = self.model.log_mel_spectrogram(audio).to(self.model.device)
    output = self.model.decode(mel, options)
    prediction = output.text.split()

    if not states.source_finished and self.continuous_write > 0:
      prediction = prediction[: self.continuous_write]

    return WriteAction(
        content=" ".join(prediction),
        finished=states.source_finished,
    )
