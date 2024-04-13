import torch
from recorder import Recorder
from transformers import WhisperProcessor, WhisperForConditionalGeneration, M2M100Tokenizer, M2M100ForConditionalGeneration
from speechbrain.inference.VAD import VAD
from transformer.decoder import HypothesisBuffer
from weights import model_names

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sample_rate = 16_000
segment_length = sample_rate * 2
context_length = sample_rate * 28
print(f'Sample rate: {sample_rate}')
print(f'Main segment: {segment_length / sample_rate} seconds')
print(f'Right context: {context_length / sample_rate} seconds')

cache_dir = 'pretrained_models'


VAD = VAD.from_hparams(
    source='speechbrain/vad-crdnn-libriparty',
    savedir=f'{cache_dir}/vad-crdnn-libriparty',
    run_opts={device: device}
)

audio_processor = WhisperProcessor.from_pretrained(
    model_names['speech2text'],
    cache_dir=cache_dir
)
speech2text = WhisperForConditionalGeneration.from_pretrained(
    model_names['speech2text'],
    cache_dir=cache_dir,
    use_safetensors=True,
    low_cpu_mem_usage=True
).to(device)

text_tokenizer = M2M100Tokenizer.from_pretrained(
    model_names['text2text'],
    cache_dir=cache_dir
)
text_tokenizer.src_lang = 'en'
text2text = M2M100ForConditionalGeneration.from_pretrained(
    model_names['text2text'],
    cache_dir=cache_dir,
    use_safetensors=True,
).to(device)

record = Recorder(sample_rate, segment_length, context_length, VAD)
buffer = HypothesisBuffer()
hypothesize = buffer is not None
trunk = 'The recording has started.'

for t, segment in enumerate(record()):
  input_features = audio_processor(
      segment,
      sampling_rate=sample_rate,
      return_tensors='pt'
  ).input_features
  output = speech2text.generate(
      input_features.to(device),
      language='english',
      task='transcribe',
      prompt_lookup_num_tokens=10,
  )

  transcription = audio_processor.decode(
      output[0],
      skip_special_tokens=True
  )
  print('Transcription:', transcription, end='\n\n')
  trunk = buffer(transcription)

  if not trunk:
    continue
  input_ids = text_tokenizer.encode(trunk, return_tensors='pt')
  outputs = text2text.generate(
      input_ids.to(device),
      forced_bos_token_id=text_tokenizer.get_lang_id('ru'))
  translation = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
  print('Trunk:', trunk, end='\n\n')
  print('Translation:', translation, end='\n\n')
