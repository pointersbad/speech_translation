from recorder import Recorder
from transformers import WhisperProcessor, WhisperForConditionalGeneration


sample_rate = 16_000
segment_length = sample_rate*3
context_length = segment_length*2

print(f"Sample rate: {sample_rate}")
print(f"Main segment: {segment_length / sample_rate} seconds")
print(f"Right context: {context_length / sample_rate} seconds")

record = Recorder(sample_rate, segment_length, context_length)
for segment in record():
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en")

    input_features = processor(
        segment, sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True)
    print(transcription[0])
