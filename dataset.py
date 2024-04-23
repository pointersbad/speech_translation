import torch
import soundfile as sf
import yaml
from itertools import groupby
from torch.utils.data import Dataset


class MuSTC(Dataset):
  def __init__(
      self, root: str,
      audio_processor,
      tokenizer,
      sample_rate,
      validation=False
  ):
    split = 'dev' if validation else 'train'
    root = f'{root}/en-ru/data/split'
    wav_root, txt_root = (f'{root}/{p}' for p in ('wav', 'txt'))
    assert root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
    with open(f'{txt_root}/{split}.yaml') as f:
      segments = yaml.load(f, Loader=yaml.BaseLoader)
    for lang in ('en', 'ru'):
      with open(f'{txt_root}/{split}.{lang}') as f:
        utterances = [r.strip() for r in f]
      for i, u in enumerate(utterances):
        segments[i][lang] = u

    self.data = []
    for wav_filename, _seg_group in groupby(segments, lambda x: x['wav']):
      wav_path = f'{wav_root}/{wav_filename}'
      sample_rate = sf.info(wav_path.as_posix()).samplerate
      seg_group = sorted(_seg_group, key=lambda x: x['offset'])
      for i, segment in enumerate(seg_group):
        offset = int(float(segment['offset']) * sample_rate)
        n_frames = int(float(segment['duration']) * sample_rate)
        self.data.append((
            wav_path.as_posix(),
            offset,
            n_frames,
            sample_rate,
            segment['en'],
            segment['ru'],
        ))
      self.audio_processor = audio_processor
      self.tokenizer = tokenizer

  def __getitem__(self, n: int):
    wav_path, offset, n_frames, sample_rate, src, tgt = self.data[n]
    waveform, _ = sf.read(
        wav_path,
        dtype='float32',
        samplerate=sample_rate,
        always_2d=True,
        frames=n_frames,
        start=offset
    )
    waveform = torch.from_numpy(waveform.T)
    src, tgt = (self.tokenizer(x) for x in (src, tgt))
    return self.audio_processor(waveform), src, tgt, sample_rate

  def __len__(self):
    return len(self.data)
