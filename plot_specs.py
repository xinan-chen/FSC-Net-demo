import os
import glob
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

AUDITION_LIKE_PARAMS = {
    8000: {
        "hop_length": 40,
        "win_length": 736,
        "n_fft": 1024,
        "n_mels": 171,
    },
    16000: {
        "hop_length": 40,
        "win_length": 1472,
        "n_fft": 2048,
        "n_mels": 341,
    },
    22050: {
        "hop_length": 56,
        "win_length": 2030,
        "n_fft": 2048,
        "n_mels": 341,
    },
    24000: {
        "hop_length": 60,
        "win_length": 2208,
        "n_fft": 4096,
        "n_mels": 682,
    },
    32000: {
        "hop_length": 80,
        "win_length": 2944,
        "n_fft": 4096,
        "n_mels": 682,
    },
    44100: {
        "hop_length": 110,
        "win_length": 4057,
        "n_fft": 4096,
        "n_mels": 682,
    },
    48000: {
        "hop_length": 120,
        "win_length": 4416,
        "n_fft": 8192,
        "n_mels": 768,
    },
}


def plot_melspec(audio_file, save_file, figsize=(16, 6), dpi=300):
        y, sr = sf.read(audio_file, dtype='float32')
        
        params = AUDITION_LIKE_PARAMS[sr]

        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            hop_length=params["hop_length"],
            win_length=params["win_length"],
            n_fft=params["n_fft"],
            n_mels=params["n_mels"],
            fmin=40,
            fmax=sr // 2,
            power=2.0,
        )
        
        # 转换为分贝单位
        S_db = librosa.power_to_db(S, ref=np.max, top_db=90)
        
        # 创建高清图形
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 使用高质量的色彩映射
        img = librosa.display.specshow(
            S_db, 
            sr=sr,
            hop_length=params["hop_length"],
            cmap='magma'  # 或者 'viridis', 'plasma', 'inferno'
        )
            
        # 隐藏所有坐标轴
        plt.axis('off')
        
        # 极致的紧凑布局
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 高质量保存
        plt.savefig(
            save_file, 
            bbox_inches='tight', 
            pad_inches=0, 
            dpi=dpi,
            facecolor='black',  # 黑色背景（可选）
            transparent=True    # 或者设置为透明背景
        )
        
        plt.close()
    

def plot_linearspec(audio_file, save_file, figsize=(16, 6), dpi=300):
        y, sr = sf.read(audio_file, dtype='float32')
     
        params = AUDITION_LIKE_PARAMS[sr]

        S = librosa.stft(
            y=y, 
            hop_length=params["hop_length"],
            win_length=params["win_length"],
            n_fft=params["n_fft"],
        )
    
        # 转换为分贝单位
        S_db = librosa.power_to_db(np.abs(S)**2, ref=np.max)
        
        # 创建高清图形
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 使用高质量的色彩映射
        img = librosa.display.specshow(
            S_db, 
            sr=sr,
            hop_length=params["hop_length"],
            cmap='magma'  # 或者 'viridis', 'plasma', 'inferno'
        )
            
        # 隐藏所有坐标轴
        plt.axis('off')
        
        # 极致的紧凑布局
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 高质量保存
        plt.savefig(
            save_file, 
            bbox_inches='tight', 
            pad_inches=0, 
            dpi=dpi,
            facecolor='black',  # 黑色背景（可选）
            transparent=True    # 或者设置为透明背景
        )
        
        plt.close()
    

def process_audio_files(input_folder, output_folder, plot_func, ext='wav', **kwargs):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    audio_files = librosa.util.find_files(input_folder, ext=ext)
    audio_files = [item for item in audio_files if f'UniPASE.{ext}' in str(item)]

    print("Find {} audio files in {}".format(len(audio_files), input_folder))
    
    for audio_file in tqdm(audio_files):
        save_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(audio_file))[0]}.png")
        # print(f"Processing {audio_file} -> {save_file}")
        plot_func(audio_file, save_file, **kwargs)
        

if __name__ == "__main__":
    input_folder = 'urgent2_wav'
    output_folder = 'urgent2_fig'
    
    process_audio_files(input_folder, output_folder, plot_melspec, ext='flac', figsize=(8, 4), dpi=100)