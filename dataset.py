import os
from torch.utils.data import Dataset
import torchaudio

config = {}


class MustCDataset(Dataset):
  def init(
      self,
      split,
      sample_rate,
      n_mels,
      tokenizer,
      pad_idx,
      eos_idx
  ):
    self.root_dir = os.path.join('data/must-c', split)
    self.audio_dir = os.path.join(self.root_dir, 'audio')
    self.text_dir = os.path.join(self.root_dir, 'text')
    self.sample_rate = sample_rate
    self.n_mels = n_mels
    self.tokenizer = tokenizer
    self.pad_idx = pad_idx
    self.eos_idx = eos_idx

  def __len__(self):
    filenames = (f.split('.')[0] for f in os.listdir(
        self.audio_dir) if f.endswith('.wav'))
    return len(filenames)

  def __getitem__(self, idx):
    filename = self.filenames[idx]
    audio_path = os.path.join(self.audio_dir, f"{filename}.wav")
    text_path = os.path.join(self.text_dir, f"{filename}.txt")
    waveform, sr = torchaudio.load(audio_path)
    if sr != self.sample_rate:
      resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
      waveform = resampler(waveform)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=self.sample_rate, n_mels=self.n_mels)(waveform)
    with open(text_path, 'r', encoding='utf-8') as f:
      text = f.readline().strip()
    tokens = self.tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        max_length=512,
        truncation=True)
    return {
        'audio': mel_spectrogram.squeeze(0),
        'text': tokens.input_ids.squeeze(0),
        'attention_mask': tokens.attention_mask.squeeze(0)
    }
