from stt import Transcriber
import subprocess
import time

def resample_single(wav_from, wav_to):
    subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
    return 0

transcriber = Transcriber(pretrain_model = 'vietnamese_wav2vec/pretrain.pt', finetune_model = 'vietnamese_wav2vec/finetune.pt',
                          dictionary = 'vietnamese_wav2vec/dict.ltr.txt',
                          lm_lexicon = 'vietnamese_wav2vec/lexicon.txt', lm_model = 'vietnamese_wav2vec/lm.bin',
                          lm_weight = 1.5, word_score = -1, beam_size = 50)
files = ['test_audio/14203.wav']
files_o = []

for i in files:
    out = i.split('.wav')[0] + '_16000.wav'
    files_o.append(out)
    resample_single(i, out)

start = time.time()
hypos = transcriber.transcribe(files_o)
end = time.time()
print('time:')
print(end - start)
