import pyaudio
import wave
import librosa
import requests
import numpy as np
from difflib import SequenceMatcher
import re

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}

def convert(text):
    """
    Convert from 'Tieng Viet co dau' thanh 'Tieng Viet khong dau'
    text: input string to be converted
    Return: string converted
    """
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        # deal with upper case
        output = re.sub(regex.upper(), replace.upper(), output)
    return output

def record():
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    return 0

with open('command.txt') as f:
    command = f.readlines()
command = [i.split('\n')[0] for i in command]

print(command)
while(1):
    _ = record()
    test_wavs, _ = librosa.load('output.wav', sr=16000, mono=True)
    test_wavs = list(np.array(test_wavs, dtype="float32").astype(float))
    datadict = {'test_wavs': test_wavs}
    r = requests.post('http://103.137.4.6:4000/stt/v1', json=datadict)
    print(r.json())
    sim = []
    for i in command:
        sim.append(similar(convert(r.json()['predict'].lower()), i))
    if max(sim) > 0.5:
        print(command[sim.index(max(sim))])



