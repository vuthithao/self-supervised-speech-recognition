from stt import Transcriber
import os
import time
from flask import Flask, jsonify
from flask import request
import json
from gevent.pywsgi import WSGIServer
import numpy as np
import librosa

app = Flask(__name__)

transcriber = Transcriber(pretrain_model = 'vietnamese_wav2vec/pretrain.pt', finetune_model = 'vietnamese_wav2vec/finetune.pt',
                          dictionary = 'vietnamese_wav2vec/dict.ltr.txt',
                          lm_lexicon = 'vietnamese_wav2vec/lexicon.txt', lm_model = 'vietnamese_wav2vec/lm.bin',
                          lm_weight = 1.5, word_score = -1, beam_size = 50)


def speechtotext():
    if request.method == "POST":
        dataDict = json.loads(request.data.decode('utf-8'))
        test_wavs = np.array(dataDict.get('test_wavs', None), dtype="float32")

    start = time.time()
    librosa.output.write_wav('output.wav', test_wavs, 16000)
    hypos = transcriber.transcribe(['output.wav'])
    end = time.time() - start
    response = jsonify({"predict": hypos[0], "time": end, "status_code": 200})
    response.status_code = 200
    response.status = 'OK'
    return response, 200

@app.route('/stt/v1', methods=['POST'])
def sttt():
    return speechtotext()

if __name__ == "__main__":
    http_server = WSGIServer(('', 4000), app)
    http_server.serve_forever()
