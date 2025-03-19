import os

import whisper

FILE_PATH = "audio.mp3"

if os.path.isfile(FILE_PATH):
    model = whisper.load_model("base")

    result = model.transcribe(FILE_PATH)

    with open("transcription.txt", "w") as file:
        file.write(result["text"])
