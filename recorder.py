from queue import Queue
from threading import Thread
from torchaudio.io import StreamReader
import time
import torch


class Recorder:
    def __init__(self, sample_rate: int, segment_length: int, context_length: int):
        super(Recorder).__init__()
        self.messages = Queue()
        self.recordings = Queue()
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.cache = ContextCacher(segment_length, context_length)

    def record(self):
        streamer = StreamReader(":0", format='avfoundation')
        streamer.add_basic_audio_stream(
            frames_per_chunk=self.segment_length, sample_rate=self.sample_rate)
        stream_iterator = streamer.stream(timeout=-1, backoff=1.0)
        while not self.messages.empty():
            (chunk,) = next(stream_iterator)
            self.recordings.put(chunk)
            time.sleep(self.segment_length/self.sample_rate)

    def __call__(self):
        print("Starting...")
        self.messages.put(True)
        record = Thread(target=self.record)
        record.start()
        while not self.messages.empty():
            chunk = self.recordings.get()
            segment = self.cache(chunk[:, 0])
            yield segment

    def stop(self):
        self.messages.get()
        print("Stopped.")


class ContextCacher:
    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(
                chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length:]
        return chunk_with_context
