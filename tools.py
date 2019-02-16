# -*- coding: utf-8 -*-

import librosa
import numpy as np
import soundfile as sf
import os


def closest_power_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                pwr = 2 ** i
                break
        if abs(pwr - target) < abs(pwr / 2 - target):
            return pwr
        else:
            return int(pwr / 2)
    else:
        return 1


class ModelConfig:
    SR = 16000                # Sample Rate
    L_FRAME = 1024            # default 1024
    L_HOP = closest_power_of_two(L_FRAME / 4)
    SEQ_LEN = 4
    # For Melspectogram
    N_MELS = 512
    F_MIN = 0.0

# Batch considered


def to_spectrogram(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav)))

# Batch considered


def to_wav(mag, phase, len_hop=ModelConfig.L_HOP):
    stft_matrix = get_stft_matrix(mag, phase)
    return np.array(list(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_matrix)))

# Batch considered


def to_wav_from_spec(stft_maxrix, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix)))

# Batch considered


def to_wav_mag_only(mag, init_phase, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP, num_iters=50):
    # return np.array(list(map(lambda m_p: griffin_lim(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p)[0], list(zip(mag, init_phase))[1])))
    return np.array(
        list(
            map(
                lambda m: lambda
                p: griffin_lim(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p),
                list(zip(mag, init_phase))[1])))

def get_magnitude(stft_matrixes):
    print(stft_matrixes.shape)
    return np.abs(stft_matrixes)

def get_phase(stft_maxtrixes):
    return np.angle(stft_maxtrixes)

def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)


def soft_time_freq_mask(target_src, remaining_src):
    mask = np.abs(target_src) / (np.abs(target_src) + np.abs(remaining_src) + np.finfo(float).eps)
    return mask

def hard_time_freq_mask(target_src, remaining_src):
    mask = np.where(target_src > remaining_src, 1., 0.)
    return mask


def write_wav(data, path, sr=ModelConfig.SR, format='wav', subtype='PCM_16'):
    sf.write('{}.wav'.format(path), data, sr, format=format, subtype=subtype)


def griffin_lim(mag, len_frame, len_hop, num_iters, phase_angle=None, length=None):
    assert(num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = get_stft_matrix(mag, phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=len_frame, hop_length=len_hop, length=length)
        if i != num_iters - 1:
            spec = librosa.stft(wav, n_fft=len_frame, win_length=len_frame, hop_length=len_hop)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = get_stft_matrix(mag, phase_angle)
    return wav


'''
to reformat each song into 4-frame batch.
Each song should have a file with all its 4-frame batch
'''


def spec_to_batch(mixed_magn, music_magn, voice_magn, filenames):

    print('filenames shape')
    print(len(filenames))
    num_wavs, freq, n_frames = mixed_magn.shape
    # Padding
    # so that each song can be divided evenly into 4 frames.
    mixed_src = pad_to_four_frame(mixed_magn)
    music_src = pad_to_four_frame(music_magn)
    voice_src = pad_to_four_frame(voice_magn)

    # prepare the folders for the outputs
    dirName1 = "./output"
    if not os.path.exists(dirName1):
        os.mkdir(dirName1)
        print("Directory ", dirName1,  " Created ")
    else:
        print("Directory ", dirName1,  " already exists")

    # prepare the sub folders
    for n in range(3):
        dirName2 = ""
        if n == 0:
            dirName2 = "./output/mixed"
        elif n == 1:
            dirName2 = "./output/music"
        elif n == 2:
            dirName2 = "./output/voice"
        if not os.path.exists(dirName2):
            os.mkdir(dirName2)
            print("Directory ", dirName2,  " Created ")
        else:
            print("Directory ", dirName2,  " already exists")

    # loop each song and separate into 4-frame batch.
    for i in range(num_wavs):
        save_filename = filenames[i].replace("dataset/", "")
        # pick out one song
        one_mixed = np.array(mixed_src[i])
        # where the dimension should be 129(batches), 4 (frames), 513 (frequencies)
        batched_one_mixed = one_mixed.T.reshape(-1, ModelConfig.SEQ_LEN, freq)
        # write to a file
        np.save("./output/mixed/"+save_filename, batched_one_mixed)

        # do the same for music and voice
        one_music = np.array(music_src[i])
        batched_one_music = one_music.T.reshape(-1, ModelConfig.SEQ_LEN, freq)
        np.save("./output/music/"+save_filename, batched_one_music)

        one_voice = np.array(voice_src[i])
        batched_one_voice = one_voice.T.reshape(-1, ModelConfig.SEQ_LEN, freq)
        np.save("./output/voice/"+save_filename, batched_one_voice)

        if i == 0:
            print(batched_one_mixed.shape)
            print(batched_one_music.shape)
            print(batched_one_voice.shape)

    pass

# def spec_to_batch_one:


def pad_to_four_frame(src):
    num_wavs, freq, n_frames = src.shape
    pad_len = 0
    if n_frames % ModelConfig.SEQ_LEN > 0:
        pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
    pad_width = ((0, 0), (0, 0), (0, pad_len))
    padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)
    assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0)
    return padded_src


def spec_to_batch_origin(src):
    num_wavs, freq, n_frames = src.shape

    # Padding
    pad_len = 0
    if n_frames % ModelConfig.SEQ_LEN > 0:
        pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
    pad_width = ((0, 0), (0, 0), (0, pad_len))
    padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

    assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0)

    batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, ModelConfig.SEQ_LEN, freq))
    return batch, padded_src
