import torch
import time
from queue import Queue
from threading import Thread
from sounddevice import InputStream


class Recorder:
  def __init__(
      self,
      sample_rate: int,
      segment_length: int,
      context_length: int,
      VAD=None
  ):
    self.messages = Queue()
    self.recordings = Queue()
    self.cache = ContextCacher(segment_length, context_length)
    self.sample_rate = sample_rate
    self.segment_length = segment_length
    self.VAD = VAD

  def record(self):
    stream = InputStream(
        self.sample_rate,
        self.segment_length,
        channels=1,
        callback=lambda x, *_: self.recordings.put(
            torch.from_numpy(x).squeeze()
        ),
    )
    with stream:
      while not self.messages.empty():
        time.sleep(self.segment_length / self.sample_rate)

  def __call__(self):
    print('Starting...')
    self.messages.put(True)
    record = Thread(target=self.record)
    record.start()
    while not self.messages.empty():
      chunk = self.recordings.get()
      if self.VAD:
        threshold = self.VAD.apply_threshold(self.VAD(chunk), 0.7, 0.3)
        silence = torch.sum(threshold) < threshold.size(1) / 4
        if silence:
          continue
      segment = self.cache(chunk)
      yield segment
      time.sleep(0.1)

  def stop(self):
    self.messages.get()
    print('Stopped.')


class ContextCacher:
  def __init__(self, segment_length: int, context_length: int):
    self.segment_length = segment_length
    self.context_length = context_length
    self.context = torch.zeros([context_length])

  def __call__(self, chunk: torch.Tensor):
    if chunk.size(0) < self.segment_length:
      pad = 0, self.segment_length - chunk.size(0)
      chunk = torch.nn.functional.pad(chunk, pad)
    chunk_with_context = torch.cat((self.context, chunk))
    self.context = chunk_with_context[-self.context_length:]
    return chunk_with_context
