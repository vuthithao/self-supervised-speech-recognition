# -*- coding: utf-8 -*-
# import os
import subprocess
import timeit
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from stt import Transcriber
import subprocess
app = Flask(__name__, template_folder='./')

transcriber = Transcriber(pretrain_model = 'vietnamese_wav2vec/pretrain.pt', finetune_model = 'vietnamese_wav2vec/finetune.pt',
                          dictionary = 'vietnamese_wav2vec/dict.ltr.txt',
                          lm_lexicon = 'vietnamese_wav2vec/lexicon.txt', lm_model = 'vietnamese_wav2vec/lm.bin',
                          lm_weight = 1.5, word_score = -1, beam_size = 50)

def resample_single(wav_from, wav_to):
    subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
    return 0

def transform(filename, filetype):
    # os.system('ffmpeg -i %s -ar 16000 -ac 1 -ab 256000 upload/upload.wav -y' % filename)
    resample_single(filename, 'temp.wav')
    subprocess.call(['ffmpeg', '-i', 'temp.wav', '-ar', '16000', '-ac', '1', '-ab', '256000', 'upload/uploaded_%s.wav' % filetype, '-y'])


def getOutput(filetype, part):
    start_time = timeit.default_timer()
    if (part == 'wavenet'):
        hypos = transcriber.transcribe(['upload/uploaded_%s.wav' % filetype])
        print(hypos)
    elapsed = timeit.default_timer() - start_time
    output = hypos[0]
    
    return [output, str(elapsed)]

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/uploadfile',methods=['POST'])
def uploadFile():
    start_time = timeit.default_timer()
    file = request.files['file']
    if len(file.filename.split('.')) < 2:
        filename = 'upload/uploaded_file.wav'
    else:
        filename = 'upload/uploaded_file.%s' % file.filename.split('.')[len(file.filename.split('.'))-1]
    file.save(filename)
    transform(filename, 'file')
    elapsed = timeit.default_timer() - start_time

    return(str(elapsed))

@app.route('/wavenetfile',methods=['POST'])
def getResWavenetFile():
    return(jsonify(result=getOutput('file', 'wavenet')))


@app.route('/uploadrecord',methods=['POST'])
def uploadRecord():
    start_time = timeit.default_timer()
    file = request.files['file']
    if len(file.filename.split('.')) < 2:
        filename = 'upload/uploaded_record.wav'
    else:
        filename = 'upload/uploaded_record.%s' % file.filename.split('.')[len(file.filename.split('.'))-1]
    file.save(filename)
    transform(filename, 'record')
    elapsed = timeit.default_timer() - start_time

    return(str(elapsed))

@app.route('/wavenetrecord',methods=['POST'])
def getResWavenetRecord():
    return(jsonify(result=getOutput('record', 'wavenet')))


if __name__ == '__main__':

    #app.run()

    # Serve the app with gevent
    http_server = WSGIServer(('', 4444), app)
    http_server.serve_forever()