#!/usr/bin/env python3
from gtts import gTTS
import os

text = "hey anki walk forward"

tts = gTTS(text=text, lang='en')
tts.save("test.mp3")

# convert to wav
os.system("ffmpeg -i test.mp3 test.wav")
