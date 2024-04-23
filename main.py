import torch
from recorder import Recorder
from speechbrain.inference.VAD import VAD
from models.hypothesis_buffer import HypothesisBuffer
from models.speech_to_text import SpeechToTextModel
from models.text_to_text import TextToTextModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sample_rate = 16_000
segment_length = sample_rate * 2
context_length = sample_rate * 28
print(f'Sample rate: {sample_rate}')
print(f'Main segment: {segment_length / sample_rate} seconds')
print(f'Right context: {context_length / sample_rate} seconds')

VAD = VAD.from_hparams(
    source='speechbrain/vad-crdnn-libriparty',
    savedir=f'pretrained_models/vad-crdnn-libriparty',
    run_opts={device: device}
)
speech2text = SpeechToTextModel(device, sample_rate)
text2text = TextToTextModel(device)

record = Recorder(sample_rate, segment_length, context_length, VAD)
buffer = HypothesisBuffer()
hypothesize = buffer is not None
trunk = 'The recording has started.'

for t, segment in enumerate(record()):
  transcription, _, _ = speech2text(segment)
  print('Transcription:', transcription, end='\n\n')
  trunk = buffer(transcription)
  if not trunk:
    continue
  translation, _, _ = text2text(trunk)
  print('Trunk:', trunk, end='\n\n')
  print('Translation:', translation, end='\n\n')
