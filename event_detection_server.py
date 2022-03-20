from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import ORJSONResponse
from panns_inference import AudioTagging, SoundEventDetection, labels
from box import Box
from pydub import AudioSegment
import pandas as pd, numpy as np, torch, librosa
import os, re

app = FastAPI()
SPEAK_EVENTS = ['speech', 'speak', 'conversation', 'male', 'female', 'narration']
SPEAK_EVENT_NAME = 'speak'
BASE = os.path.dirname(__file__) 
TEMP_FOLDER = BASE+'/temp/'
EVENT_THRESHOLD = 0.1

args  = Box({
    'sample_rate': 32000,
    'window_size': 1024,
    'hop_size': 320,
    'mel_bins' : 64,
    'fmin' : 50,
    'fmax': 14000,
    'model_type': 'Cnn14_DecisionLevelMax',
    'checkpoint_path': f'{BASE}/Cnn14_DecisionLevelMax_mAP=0.385.pth',
    # 'audio_path' : path,
    'cuda': False
})

@app.on_event("startup")
async def startup():
    global sed_model
    sed_model = SoundEventDetection(checkpoint_path=f'{BASE}/Cnn14_DecisionLevelMax_mAP=0.385.pth', device='cuda:1')

@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    '''return error code for debugging'''
    import traceback
    content = "".join(
        traceback.format_exception(etype=type(
            exc), value=exc, tb=exc.__traceback__)
    )
    print(content)
    return ORJSONResponse(status_code=500, content=content, media_type='text/plain')


@app.post('/detect_event', response_class=ORJSONResponse)
def audio_event_detection(file: UploadFile = File(...), threshold:float=EVENT_THRESHOLD ,detail:bool=False, background_tasks: BackgroundTasks = None):
    assert file.content_type.split('/')[0] == 'audio'
    format = file.content_type.split('/')[-1]
    audio = AudioSegment.from_file(file.file, format=format).set_frame_rate(32000)
    assert audio.duration_seconds < 60, f'{file.filename} duration too long: {audio.duration_seconds}'
    temp_file = TEMP_FOLDER+file.filename+'.mp3'
    audio.export(temp_file, format='mp3')
    # inference
    (audio, _) = librosa.core.load(temp_file, sr=32000, mono=True)
    audio = audio[None, :]
    framewise_output = sed_model.inference(audio).squeeze()
    background_tasks.add_task(os.remove, temp_file)
    # top 10 events
    probs_mean = framewise_output.mean(axis=0)
    top10 = {labels[i]: probs_mean[i] for i in np.argsort(-probs_mean)[:10]}
    print('Top 10 events:\n', top10)
    # event_series
    top_events = extract_event(framewise_output, labels)
    top_events.label = top_events.label.apply(lambda x: SPEAK_EVENT_NAME if any(s.lower() in x.lower() for s in SPEAK_EVENTS) else x)
    # detect male and female
    male_idx = [i for i,l in enumerate(labels) if re.findall('^male\s', l, re.I)]
    female_idx = [i for i,l in enumerate(labels) if re.findall('^female\s', l, re.I)]
    male_prob = framewise_output[:, male_idx].sum()/framewise_output.shape[0]
    female_prob = framewise_output[:, female_idx].sum()/framewise_output.shape[0]
    sum_mf = male_prob+female_prob
    if sum_mf > 0:
        male_prob, female_prob = male_prob/sum_mf, female_prob/sum_mf
    else:
        male_prob, female_prob = 0, 0
    speak_prob = top_events.query('label==@SPEAK_EVENT_NAME').prob.sum() / top_events.query('prob>@EVENT_THRESHOLD').shape[0]
    events = top_events.query('prob>@threshold').label.unique().tolist()
    result = {
        'speak': speak_prob.item(),
        'male': male_prob.item(),
        'female': female_prob.item(),
        'events': events,
        'file': file.filename
    }
    if detail:
        result['detail'] = top_events.to_dict(orient='list')
    return result

def extract_event(probs, labels):
    '''extract event from probs
    '''
    top_idx = np.argsort(-probs, axis=1)[:,0]
    top_probs = probs[range(len(probs)), top_idx]
    top_labels = [labels[i] for i in top_idx]
    top_events = pd.DataFrame({'label': top_labels, 'prob': top_probs})
    return top_events


def to_python(dict_):
    '''convert dict to python object
    '''
    return {k: v.items() if isinstance(v, np.float64) else v for k, v in dict_.items()}

if __name__ == '__main__':
    '''uvicorn audio_event_detection_server:app --workers=4 --host=0.0.0.0 --port=9012'''
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9012, debug=True)

    # from glob import glob
    # import asyncio
    # asyncio.run(startup())
    # paths = glob(f'{BASE}/sample/脉冲/*.m4a', recursive=False)
    # for path in paths:
    #     file = UploadFile(file=open(path, 'rb'), filename=os.path.basename(path), content_type='audio/m4a')
    #     print(audio_event_detection(file))
