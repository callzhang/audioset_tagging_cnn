#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audio event detection script
   1) check if file are longer than 600s (due to linearity of GPU RAM used to the model)
   2) slice it if necessary
   3) do the SED in pytorch, parameter are in the code
   4) extract event probability in each frame, using keywords matching
   5) save it and loop to the next
"""

from box import Box
from glob import glob
import pickle, random, hashlib, os, time, datetime, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append('pytorch')
from inference import audio_tagging, sound_event_detection
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from retry import retry

chunk_length = 600  # 10min
kw = ['speech', 'speak', 'conversation', 'male', 'female']
nw = ['music']

sed_cache = '/mnt/nas/中科大44国语言/sed_cache/'

def get_all_mp3():
    paths = glob('/mnt/nas/中科大44国语言/希伯来语/YouTube/*.mp3')
    paths += glob('/mnt/nas/中科大44国语言/印地语/YouTube/*.mp3')
    paths += glob('/mnt/nas/中科大44国语言/缅甸语/YouTube/*.mp3')
    paths += glob('/mnt/nas/中科大44国语言/格鲁吉亚/YouTube/*.mp3')
    paths += glob('/mnt/nas/中科大44国语言/test/*.mp3')
    return paths

def get_all_wav():
    root = 'sample/'
    lang_folder = glob(root+'**/*.mp3')
    lang_folder = [l for l in lang_folder if os.path.exists(l+'valid_audio.txt')]
    audio_files = []
    for folder in lang_folder:
        with open(folder+'valid_audio.txt', 'r') as f:
            paths = f.read().split('\n')
        for path in paths:
            audio_folder = folder+path
            wav_files = glob(audio_folder+'/*.wav')
            audio_files += wav_files
    return audio_files

def get_md5(path):
    md5_hash = hashlib.md5()
    with open(path, "rb") as f:
        content = f.read()
    md5_hash.update(content)
    digest = md5_hash.hexdigest()
    return digest

def is_cached(path):
    hash = get_md5(path)
    if os.path.exists(sed_cache+hash+'.csv') and os.path.exists(sed_cache+hash+'.png'):
        return True
    else:
        return False

# fix file and corruption (caused by me)
@retry(tries=2, delay=2)
def slice_audio(path):
    if os.path.getsize(path) > 512*1024**2:
        print(f'file too large: {path}, SKIP!')
        return
    audio = AudioSegment.from_file(path)
    duration = audio.duration_seconds
    if duration > chunk_length+5:  # tolerance
        print(f'{path} has duration of {duration:.1f}s, slicing...')
        n_split = int(duration/chunk_length)+1
        for i in range(n_split):
            audio_slice = audio[i*chunk_length*1000:(i+1)*chunk_length*1000]
            slice_name = path.replace('.mp3', f'({i}).mp3')
            audio_slice.export(slice_name, format='mp3')
            print(f'{slice_name} saved')

        new_folder = os.path.dirname(path)+'/too_large_sliced/'
        os.makedirs(new_folder, exist_ok=True)
        os.rename(path, new_folder+os.path.basename(path))
        print(f'{path} moved to "too_large_sliced"')
        return 
    elif duration < 5:  # 5s
        print(f'{path} removed with duration:{duration:.1f}s')
        os.remove(path)
    # check stats file
    if os.path.exists(path+'.csv'):
        stats = pd.read_csv(path+'.csv')
        sd = stats.index.max()/100
        if int(sd) != int(duration):
            print(f'{path} stats file corrupted: {sd} vs {duration}')
            os.remove(path+'.csv')
    return 1


# args  = Box({
#     'sample_rate' : 16000,
#     'window_size' : 512,
#     'hop_size' : 160,
#     'mel_bins' : 64,
#     'fmin' : 50,
#     'fmax' : 8000,
#     'model_type' : 'Cnn14_16k',
#     'checkpoint_path' : 'Cnn14_16k_mAP=0.438.pth',
#     'audio_path' : path,
#     'cuda': True
# })

# print(f'starting audio tagging:{path}')
# clipwise_output, labels = audio_tagging(args)
# print('clipwise_output\n', clipwise_output)
# print('labels\n', labels)

def audio_event_detection(path):
    '''audio event 
    '''
    args2 = Box({
    'sample_rate': 32000,
    'window_size': 1024,
    'hop_size': 320,
    'mel_bins': 64,
    'fmin': 50,
    'fmax': 14000,
    'model_type': 'Cnn14_DecisionLevelMax',
    'checkpoint_path': 'Cnn14_DecisionLevelMax_mAP=0.385.pth',
    'audio_path': path,
    'cuda': True
})
    print(f'starting SED:{path}')
    if is_cached(path):
        print(f'{path} cached, skip!')
        return
    # d = AudioSegment.from_file(path).duration_seconds,    
    # # if d > (chunk_length+5):,    
    # #     print(f'{path} too long: {d}s, skip!'),    
    # #     continue

    #run SED prediction
    framewise_output, labels = sound_event_detection(args2)

    def analyze_frame(probs, positive_keys, negative_keys):
        top5 = np.argsort(-probs)[:5]
        events = {labels[i]: probs[i] for i in top5}
        speak = [p for e, p in events.items() for k in positive_keys if k in e.lower()]
        noise = [p for e, p in events.items() if all([k not in e.lower() for k in positive_keys])]
        music = [p for e, p in events.items() for k in negative_keys if k in e.lower()]
        male = [p for e, p in events.items() if 'male' in e.lower()]
        female = [p for e, p in events.items() if 'female' in e.lower()]
        stats = {
            'speak': sum(speak),
            'noise': sum(noise),
            'music': sum(music),
            'male': sum(male),
            'female': sum(female)
        }
        return stats
    
    probs = framewise_output.mean(axis=0)
    top10 = {labels[i]: probs[i] for i in np.argsort(-probs)[:10]}
    print('Top 10 events:\n', top10)
    selection = []
    stats_all = pd.DataFrame([analyze_frame(f, kw, nw) for f in framewise_output])
    size = 12000 #120s window size
    n = int(stats_all.shape[0]/size)+1
    fig = plt.figure()
    speech_avg = stats_all['speak'].quantile(0.5)
    speech_10 = stats_all['speak'].quantile(0.1)
    for i in range(n):
        ax = fig.add_subplot(n, 1, i+1)
        stats = stats_all[i*size:(i+1)*size]
        min_idx = stats.index[0]
        max_idx = stats.index[-1]
        if min_idx == max_idx:
            continue
        stats.plot(ax=ax, figsize=[20, 5*n])
        ax.axline((min_idx, speech_10), (max_idx, speech_10),
                label='speak low', linestyle='--')
        ax.axline((min_idx, speech_avg), (max_idx, speech_avg),
                label='speak avg', linestyle=':')
    
    #save
    hash = get_md5(path)
    fig.savefig(sed_cache + hash +'.png')
    stats_all.to_csv(sed_cache + hash+'.csv')


if __name__ == "__main__":
    # while True:
    print(f'Starting scan at {datetime.datetime.now()}')
    # paths = get_all_mp3()
    # 检查语言长度
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     result = list(tqdm(executor.map(slice_audio, paths), total=len(paths)))
    
    # SED
    # paths = get_all_mp3()

    # paths = get_all_wav()
    paths = glob('sample/**/*.mp3', recursive=True)
    for path in tqdm(paths):
    # for path in tqdm(paths[::-1]):
    # for path in tqdm(paths[int(len(paths)/3):int(len(paths)/3*2)]):
        audio_event_detection(path)
    print(f'Finished SED at {datetime.datetime.now()}')
    # sleep
    time.sleep(3600)
