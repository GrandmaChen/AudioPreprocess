# -*- coding: utf-8 -*-

import librosa
import numpy as np
import tools as tools
from os import walk

PATH = 'dataset'
SAMPLE_RATE = 16000
DURATION = 8.192
TOTAL_AMOUNT_SAMPLES = int(SAMPLE_RATE * DURATION)

allWavFiles = []
filenames_array = []
# start from here


def _sample_range(wav):

    # wav_len = wav.shape[-1]
    # This is good however bad for consistency in testing.
    # start = np.random.choice(range(np.maximum(1, wav_len - TOTAL_AMOUNT_SAMPLES)), 1)[0]
    start = 0
    end = start + TOTAL_AMOUNT_SAMPLES

    #　如果是单轨
    if wav.ndim == 1:
        wav = wav[start:end]

    # 两轨的话
    else:
        wav = wav[:, start:end]
    return wav

# Pad 到 duration 那么长的时间段


def _pad_wav(wav):

    pad_len = np.maximum(0, TOTAL_AMOUNT_SAMPLES - wav.shape[-1])
    if wav.ndim == 1:
        pad_width = (0, pad_len)
    else:
        pad_width = ((0, 0), (0, pad_len))
    wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)
    return wav


'''
Read the file names, and returns mixed, human voice, and music, as raw_wav
Parameters
    ----------
    filenames : array of audio file names

    Returns
    -------
    mixed: mixed music and human voice
    music_raw: music alone
    voice_raw: voice alone
'''


def extract(filenames):
    # load wav -> pad if necessary to fit sr*sec -> get random samples with len = sr*sec -> map = do this for all in filenames -> put in np.array
    resultArray = []
    for file in filenames:
        raw_wav = librosa.load(file, sr=SAMPLE_RATE, mono=False)[0]
        # 看看是不是两轨
        assert(raw_wav.ndim <= 2)

        padded_wav = _pad_wav(raw_wav)
        # We used to take only 8.192 sec from each song, now the song length can be different
        # Nah we changed it back.
        # sampled_wav = padded_wav
        sampled_wav = _sample_range(padded_wav)
        resultArray.append(sampled_wav)
        
    all_raw = np.array(resultArray)
    # Above is equivalent to :
    # src1_src2 = np.array(list(map(lambda f: _sample_range(
    #     _pad_wav(librosa.load(f, sr=SAMPLE_RATE, mono=False)[0])), filenames)))

    # mixing music and voice into one channel
    mixed = np.array(list(map(lambda f: librosa.to_mono(f), all_raw)))
    music_raw, voice_raw = all_raw[:, 0], all_raw[:, 1]
    return mixed, music_raw, voice_raw


def load_wav():
    allWavFiles = []
    for (root, dirs, files) in walk(PATH):
        allWavFiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])

    allWavFiles = sorted(allWavFiles)
    # filenames_array = filenames
    mixed, music_raw, voice_raw = extract(allWavFiles)

    return mixed, music_raw, voice_raw, allWavFiles


def convert_raw_audio(mixed, music_raw, voice_raw, filenames):
    print("mixed raw shape: " + str(mixed.shape))
    print("music raw shape: " + str(music_raw.shape))
    print("voice raw shape: " + str(voice_raw.shape))
    mixed_spec = tools.to_spectrogram(mixed)  # undergo STFT
    mixed_magn = tools.get_magnitude(mixed_spec)  # convert magnitude

    print("mixed_spec shape: " + str(mixed_spec.shape))
    print("mixed_magn shape: " + str(mixed_magn.shape))

    music_spec, voice_spec = tools.to_spectrogram(music_raw), tools.to_spectrogram(voice_raw)
    music_magn, voice_magn = tools.get_magnitude(music_spec), tools.get_magnitude(voice_spec)

    print("music_spec shape: " + str(music_spec.shape))
    print("music_magn shape: " + str(music_magn.shape))
    print("voice_spec shape: " + str(voice_spec.shape))
    print("voice_magn shape: " + str(voice_magn.shape))

    # now shape of spec and magn are all (980,513,513)
    # where 980 is number of song
    # 513,513 is frequency bines and time frames respectively
    # Therefore each time step of RNN we should input one row of the last dimension
    # Correct me if I am wrong.

    tools.spec_to_batch(mixed_magn, music_magn, voice_magn, filenames)


def main():
    mixed, music, voice, filenames = load_wav()
    convert_raw_audio(mixed, music, voice, filenames)


if __name__ == "__main__":
    main()
