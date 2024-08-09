# -*- coding: utf-8 -*-
import os
from pesq import pesq
from scipy.io import wavfile
clean_folder = "/conformer/data/wsj_8k/test/"
processed_folder = "/conformer/data/em_two_stage/wav2ibm/adpcm16kbps/"
output_file = "pesq_results.txt"

# 檢查文件夾是否存在
if not os.path.exists(clean_folder) or not os.path.exists(processed_folder):
    raise FileNotFoundError("not")

# 獲取文件名列表
clean_files = sorted([f for f in os.listdir(clean_folder) if f.endswith('.wav')])
processed_files = sorted([f for f in os.listdir(processed_folder) if f.endswith('.wav')])

# 確認經過處理的文件數是乾淨文件數的兩倍
if len(processed_files) != 2 * len(clean_files):
    raise ValueError("two")

# 計算PESQ值
with open(output_file, 'w') as out_file:
    for clean_file in clean_files:
        clean_path = os.path.join(clean_folder, clean_file)
        
        # 找到對應的兩個處理後的文件
        processed_file1 = os.path.join(processed_folder, clean_file.replace(".wav", "_ibm.wav"))
        processed_file2 = os.path.join(processed_folder, clean_file.replace(".wav", ".wav"))  # 根據具體命名規則進行替換

        # 讀取音檔
        rate, ref = wavfile.read(clean_path)
        rate, deg1 = wavfile.read(processed_file1)
        rate, deg2 = wavfile.read(processed_file2)
        
        # 計算PESQ
        pesq_score1 = pesq(rate, ref, deg1, 'nb')
        pesq_score2 = pesq(rate, ref, deg2, 'nb')
        
        out_file.write(f"{clean_file} vs {processed_file1}: {pesq_score1}\n")
        out_file.write(f"{clean_file} vs {processed_file2}: {pesq_score2}\n")
  # 計算PESQ平均值
    avg_pesq_score1 = sum(pesq_score1) / len(pesq_score1)
    avg_pesq_score2 = sum(pesq_score2) / len(pesq_score2)

    out_file.write(f"\nAverage PESQ Score for _ibm files: {avg_pesq_score1}\n")
    out_file.write(f"Average PESQ Score for _other files: {avg_pesq_score2}\n")
print(f"PESQ {output_file}")