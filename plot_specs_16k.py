import os
import glob
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def plot_melspec(audio_file, save_file, n_mels=256, hop_length=128, n_fft=1024, figsize=(16, 6), dpi=300):
        y, sr = sf.read(audio_file, dtype='float32')
            
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            fmax=sr//2,
            power=2.0
        )
        
        # 转换为分贝单位
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # 创建高清图形
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 使用高质量的色彩映射
        img = librosa.display.specshow(
            S_db, 
            sr=sr,
            hop_length=hop_length,
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
    

def plot_linearspec(audio_file, save_file, n_mels=256, hop_length=128, n_fft=1024, figsize=(16, 6), dpi=300):
        y, sr = sf.read(audio_file, dtype='float32')
            
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=hop_length*2)
    
        # 转换为分贝单位
        S_db = librosa.power_to_db(np.abs(S)**2, ref=np.max)
        
        # 创建高清图形
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 使用高质量的色彩映射
        img = librosa.display.specshow(
            S_db, 
            sr=sr,
            hop_length=hop_length,
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
    

def process_audio_files(input_folder, output_folder, plot_func, ext='wav'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    audio_files = librosa.util.find_files(input_folder, ext=ext)
    print("Find {} audio files in {}".format(len(audio_files), input_folder))
    
    for audio_file in tqdm(audio_files):
        save_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(audio_file))[0]}.png")
        # print(f"Processing {audio_file} -> {save_file}")
        plot_func(audio_file, save_file)
        

if __name__ == "__main__":
    input_folder = 'audio_examples'
    output_folder = 'audio_figs'
    
    process_audio_files(input_folder, output_folder, plot_melspec)