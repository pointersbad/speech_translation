import torch
import numpy as np
from recorder import Recorder
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from speechbrain.inference.VAD import VAD
from transformer.decoder import HypothesisBuffer

# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'

sample_rate = 16_000
segment_length = sample_rate * 3
context_length = sample_rate * 12
hypothesize = True

model_name = 'openai/whisper-small'
cache_dir = 'pretrained_models'

print(f'Sample rate: {sample_rate}')
print(f'Main segment: {segment_length / sample_rate} seconds')
print(f'Right context: {context_length / sample_rate} seconds')

VAD = VAD.from_hparams(
    source='speechbrain/vad-crdnn-libriparty',
    savedir=f'{cache_dir}/vad-crdnn-libriparty'
).to(device)
processor = WhisperProcessor.from_pretrained(
    model_name,
    cache_dir,
)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation='eager',
    cache_dir=cache_dir,
    use_safetensors=True
).to(device)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language='russian',
    task='translate'
)
model = torch.compile(model)
print(model)

record = Recorder(sample_rate, segment_length, context_length, VAD)
buffer = HypothesisBuffer()
trunk = ''

for i, segment in enumerate(record()):
  prompt_ids = processor.get_prompt_ids(trunk) if hypothesize else None
  forgotten = max(i - (context_length + segment_length) / segment_length, 0)
  input_features = processor(
      segment[max(int((buffer.offset - forgotten) * segment_length), 0):],
      sampling_rate=sample_rate,
      return_tensors='pt'
  ).input_features
  predicted_ids = model.generate(
      input_features.to(device),
      repetition_penalty=1.5,
      renormalize_logits=True,
      num_beams=2,
      prompt_ids=prompt_ids,
      return_token_timestamps=hypothesize
  )

  if hypothesize:
    ids = predicted_ids.sequences.numpy().astype(np.uint16)
    timestamps = predicted_ids.token_timestamps.numpy().astype(np.float32)
    timestamps = timestamps / (timestamps.max() or 1)
    timestamped_ids = np.array([x.squeeze()for x in (ids, timestamps)]).T
    timestamped_ids = filter(
        lambda x: x[0] not in processor.tokenizer.all_special_ids,
        timestamped_ids
    )
    hypothesis = processor.decode(buffer([*timestamped_ids]))
    trunk += hypothesis
    print(trunk)

  transcription = processor.batch_decode(
      predicted_ids.sequences if hypothesize else predicted_ids,
      skip_special_tokens=True
  )
  print(transcription[0])
