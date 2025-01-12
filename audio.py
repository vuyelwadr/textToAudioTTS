from TTS.api import TTS

# Initialize TTS with the desired model
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=False
)

# Load text from the file
with open('extracted_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Convert text to speech and save as an audio file
tts.tts_to_file(text=text, file_path="extracted_text.wav", speaker="Tammie Ema", language="en")
# ch1 took 2069.9454407691956 0.586448923755413
# ch2 Processing time: 1562.6136786937714  > Real-time factor: 0.5454422665793309
# ch3 Processing time: 1202.6651329994202 Real-time factor: 0.5322818294180808
# 4 to end Processing time: 17205.384893894196 Real-time factor: 0.6089975354642111