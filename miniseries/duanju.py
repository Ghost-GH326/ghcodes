import subprocess
import re
from datetime import timedelta

# 输入一段文本，按照标点符号划分，添加到list中
input_text = "Apakah Anda kekurangan uang akhir-akhir ini Akulaku bias pinjam dana hingga 15 juta lho. Bahkan jika Anda tidak memerlukannya sekarang, Anda dapat mendaftar dan mengajukan terlebih dahulu, dan saat memerlukan, cukup pilih menggunakan pembayaran Akulaku dan mengajukan dengan satu klik, dana bisa masuk dalam waktu, cepat secepat, 30 detik. Selain itu, bunga Akulaku rendah hingga 003%, dan juga menyediakan layanan cicilan, hingga 12 kali, untuk mengurangi tekanan pembayaran. Yang paling penting, Sudah berizin resmi dan diawasi OJK,menjamin semua proses pinjaman adalah legal dan sesuai regulasi, Anda bisa menggunakan dengan aman Jika Anda membutuhkan dana sekarang, jangan ragu lagi, klik tautan di bawah, daftar dengan nomor telepon dan informasi identitas Anda sekarang"
sentences = re.split(r'[,.!?]', input_text)
sentences = [s.strip() for s in sentences if s.strip()]

# 使用FFmpeg根据音量识别音频停顿
result = subprocess.check_output(['ffmpeg', '-i', '/home/luoqin705/jieya/印尼音频1.25/1_1.mp3', '-af', 'silencedetect=noise=-30dB:d=0.15', '-f', 'null', '-'], stderr=subprocess.STDOUT, text=True)

# 解析音量信息并生成时间码
subtitle_lines = []
start_time = 0
end_time = 0
threshold = 0  # 设置音量阈值，根据具体需求调整

for line in result.split('\n'):
    match = re.search(r"silence_start: (\d+\.\d+)", line)
    if match:
        start_time = float(match.group(1))
    match = re.search(r"silence_end: (\d+\.\d+)", line)
    if match:
        end_time = float(match.group(1))
        if end_time - start_time > 0:  # 设置最小停顿时间为0.5秒
            start_time_str = str(timedelta(seconds=start_time))
            end_time_str = str(timedelta(seconds=end_time))
            subtitle_lines.append(f"0,{start_time_str[:-4]}, {end_time_str[:-4]}")

# 打印划分后的句子和音频停顿时间码
print(len(sentences)==len(subtitle_lines))
print(len(sentences),len(subtitle_lines))
print(sentences)

a = []
for ids in range(1,len(subtitle_lines)):
    # print(ids)
    start_t = subtitle_lines[ids-1].split(", ")[1]
    end_t = subtitle_lines[ids].split(", ")[1]
    b = f'Dialogue: 0,{start_t},{end_t},Default,,0,0,0,,{{\\fad(1,1)}}{sentences[ids]}'
    print(b)
    with open("output.srt","a") as f:
        f.write(b+"\n")
# print("字幕时间码",a)
