import whisper
import torchaudio
import math

# 加载 Whisper 模型
model = whisper.load_model("medium")

# 定义函数，用于处理音频片段
def process_audio_segment(audio_segment):
    # 将音频转换为对数梅尔频谱图，并移动到 CPU 上
    mel = whisper.log_mel_spectrogram(audio_segment).to("cpu")

    # 解码音频
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return result.text

# 加载完整的音频文件
audio_path = "/home/guohao826/video_1.wav"
audio, sample_rate = torchaudio.load(audio_path)

# 计算音频的长度（以帧为单位）
audio_length_frames = audio.size(1)

# 将音频分割为 30 秒的片段
segment_length_frames = 30 * sample_rate
num_segments = math.ceil(audio_length_frames / segment_length_frames)

# 处理每个片段
results = []
for i in range(num_segments):
    start_frame = i * segment_length_frames
    end_frame = min((i + 1) * segment_length_frames, audio_length_frames)
    audio_segment = audio[:, start_frame:end_frame]
    result_text = process_audio_segment(audio_segment)
    results.append(result_text)

# 合并结果
full_text = "\n".join(results)
print(full_text)