from inference import audio_tagging, sound_event_detection
from box import Box
from glob import glob
import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from retry import retry

paths = glob('/mnt/nas/中科大44国语言/希伯来语/YouTube/*.mp3')
paths += glob('/mnt/nas/中科大44国语言/印地语/YouTube/*.mp3')
paths += glob('/mnt/nas/中科大44国语言/缅甸语/YouTube/*.mp3')
paths += glob('/mnt/nas/中科大44国语言/格鲁吉亚/YouTube/*.mp3')

chunk_length = 600  # 10min

# def slice_audio(path):


@retry(tries=2, delay=2)
def slice_audio(path):
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
    elif duration < 5:  # 5s
        print(f'{path} removed with duration:{duration:.1f}s')
        os.remove(path)

# 检查语言长度
# with ThreadPoolExecutor(max_workers=10) as executor:
#     result = list(tqdm(executor.map(slice_audio, paths), total=len(paths)))

print('----->audio sliced!')
paths = glob('/mnt/nas/中科大44国语言/希伯来语/YouTube/*.mp3')
paths += glob('/mnt/nas/中科大44国语言/印地语/YouTube/*.mp3')
paths += glob('/mnt/nas/中科大44国语言/缅甸语/YouTube/*.mp3')
paths += glob('/mnt/nas/中科大44国语言/格鲁吉亚/YouTube/*.mp3')


kw = ['speech', 'speak', 'conversation']
nw = ['music']

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

for path in tqdm(paths):
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
    if os.path.exists(path+'.png'):
        print(f'{path} already procedded, skip')
        continue
    d = AudioSegment.from_file(path).duration_seconds
    if d > (chunk_length+5):
        print(f'{path} too long: {d}s, skip!')
        continue
    framewise_output, labels = sound_event_detection(args2)

    def analyze_frame(probs, positibe_keys, negative_keys):
        top5 = np.argsort(-probs)[:5]
        events = {labels[i]: probs[i] for i in top5}
        speak = [p for e, p in events.items() for k in positibe_keys if k in e.lower()]
        noise = [p for e, p in events.items() if all([k not in e.lower() for k in positibe_keys])]
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
    stats_all = pd.DataFrame([analyze_frame(f, kw, nw)
                              for f in framewise_output])
    size = 12000
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
    fig.savefig(path+'.png')
    stats_all.to_csv(path+'.csv')
