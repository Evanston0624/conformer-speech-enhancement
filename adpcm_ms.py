import subprocess
import os
def encode_to_adpcm(input_wav, output_adpcm, pcm_code, bitrate='32k', sample_rate=8000, channels=1):
    command = [
        'ffmpeg',
        '-y',
        '-i', input_wav,
        '-c:a', pcm_code,
        '-b:a', bitrate,
        '-ar', str(sample_rate),
        '-ac', str(channels),
        output_adpcm
    ]
    subprocess.run(command, check=True)

def decode_from_adpcm(input_adpcm, output_wav, sample_rate=8000, channels=1):
    command = [
        'ffmpeg',
        '-y',
        '-i', input_adpcm,
        '-c:a', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', str(channels),
        output_wav
    ]
    subprocess.run(command, check=True)

if __name__ == '__main__':
    pcm_code = 'adpcm_g726'
    # adpcm_ms：Microsoft ADPCM
    # adpcm_ima_wav：IMA ADPCM in WAV files
    # adpcm_yamaha：Yamaha ADPCM
    # adpcm_g722：G.722 ADPCM
    # adpcm_g726：G.726 ADPCM
    fs = 8000
    chn_num = 1
    bitrates = ['32k', '64k']
    for bitrate in bitrates :
        data_path = os.path.join('data', 'test', pcm_code)
        input_wav = 'input.wav'

        output_adpcm = os.path.join(data_path, f'output_{bitrate}_adpcm.wav')
        output_wav = os.path.join(data_path, f'output_{bitrate}_decoded.wav')

        # 编码为 ADPCM，调整参数
        encode_to_adpcm(input_wav, output_adpcm, pcm_code=pcm_code, bitrate=bitrate, sample_rate=fs, channels=chn_num)

        # 解码为 WAV，调整参数
        decode_from_adpcm(output_adpcm, output_wav, sample_rate=fs, channels=chn_num)