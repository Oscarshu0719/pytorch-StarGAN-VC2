import argparse
import glob
import librosa
import numpy as np
import os
import pyworld as pw
from utility import *
from datetime import datetime

FEATURE_DIM = 36
SAMPLE_RATE = 16000
FRAMES = 128
FFTSIZE = 1024
SPEAKERS_NUM = len(speakers)
CHUNK_SIZE = 1
EPSILON = 1e-10
MODEL_NAME = 'stargan-vc2'

def load_wavs(dataset: str, sr):
    """
        `data`: contains all audios file path. 
        `resdict`: contains all wav files.   
    """

    data = {}
    with os.scandir(dataset) as it:
        for entry in it:
            if entry.is_dir():
                data[entry.name] = []
                with os.scandir(entry.path) as it_f:
                    for onefile in it_f:
                        if onefile.is_file():
                            data[entry.name].append(onefile.path)
    print(f'* Loaded keys: {data.keys()}')
    resdict = {}

    cnt = 0
    for key, value in data.items():
        resdict[key] = {}

        for one_file in value:
            filename = one_file.split('/')[-1].split('.')[0] 
            newkey = f'{filename}'
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)
            y,_ = librosa.effects.trim(wav, top_db=15)
            wav = np.append(y[0], y[1: ] - 0.97 * y[: -1])

            resdict[key][newkey] = wav
            print('.', end='')
            cnt += 1

    print(f'\n* Total audio files: {cnt}.')
    return resdict

def chunks(iterable, size):
    """
        Yield successive n-sized chunks from iterable.
    """

    for i in range(0, len(iterable), size):
        yield iterable[i: i + size]

def wav_to_mcep_file(dataset: str, sr=SAMPLE_RATE, processed_filepath: str='./data/processed'):
    """
        Convert wavs to mcep feature using image repr.
    """

    shutil.rmtree(processed_filepath)
    os.makedirs(processed_filepath, exist_ok=True)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    print(f'* Total audio files: {allwavs_cnt}.')

    d = load_wavs(dataset, sr)
    for one_speaker in d.keys():
        values_of_one_speaker = list(d[one_speaker].values())
       
        for index, one_chunk in enumerate (chunks(values_of_one_speaker, CHUNK_SIZE)):
            wav_concated = [] 
            temp = one_chunk.copy()

            for one in temp:
                wav_concated.extend(one)
            wav_concated = np.array(wav_concated)

            f0, ap, sp, coded_sp = cal_mcep(wav_concated, sr=sr, dim=FEATURE_DIM)
            newname = f'{one_speaker}_{index}'
            file_path_z = os.path.join(processed_filepath, newname)
            np.savez(file_path_z, f0=f0, coded_sp=coded_sp)
            print(f'[SAVE]: {file_path_z}')

            for start_idx in range(0, coded_sp.shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = coded_sp[:, start_idx : start_idx+FRAMES]

                if one_audio_seg.shape[1] == FRAMES:
                    temp_name = f'{newname}_{start_idx}'
                    filePath = os.path.join(processed_filepath, temp_name)
                    np.save(filePath, one_audio_seg)
                    print(f'[SAVE]: {filePath}.npy')
            
def world_features(wav, sr, fft_size, dim):
    f0, timeaxis = pw.harvest(wav, sr)
    sp = pw.cheaptrick(wav, f0, timeaxis, sr,fft_size=fft_size)
    ap = pw.d4c(wav, f0, timeaxis, sr, fft_size=fft_size)
    coded_sp = pw.code_spectral_envelope(sp, sr, dim)

    return f0, timeaxis, sp, ap, coded_sp

def cal_mcep(wav, sr=SAMPLE_RATE, dim=FEATURE_DIM, fft_size=FFTSIZE):
    """
        Calculate MCEPs given wav singnal.
    """

    f0, timeaxis, sp, ap, coded_sp = world_features(wav, sr, fft_size, dim)
    coded_sp = coded_sp.T

    return f0, ap, sp, coded_sp

if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser(description='Convert the wav waveform to mel-cepstral coefficients(MCCs)\
    and calculate the speech statistical characteristics.')
    
    input_dir = './data/spk'
    output_dir = './data/processed'
   
    parser.add_argument('--input_dir', type=str, default=input_dir, help='Directory of input data.')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Directory of processed data.')
    
    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    wav_to_mcep_file(input_dir, SAMPLE_RATE, processed_filepath=output_dir)

    generator = GenerateStatistics(output_dir)
    generator.generate_stats()
    generator.normalize_dataset()
    end = datetime.now()
    
    print(f"* Duration: {end-start}.")
