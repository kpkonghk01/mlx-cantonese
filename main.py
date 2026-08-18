import numpy as np
from scipy.io import wavfile
from mlx_audio.music import load

def main():
    model_id = "mlx-community/MiniMax-Music3-nvfp4"
    print(f"正在載入模型: {model_id}...")
    
    # 根據 Model Card 正確導入方式
    model = load(model_id)

    # 風格設定（廣東話獨白、溫柔女聲、鋼琴）
    text = "Cantonese monologue, soft female vocal, clear speech, no background music"
    
    # 歌詞/獨白文本（加入 [Spoken] 標籤控制為說話而非唱歌）
    lyrics = "[Spoken]\n落雨嘅日子，總係會令人諗起好多過去嘅片段。\n聽住雨聲，好似時間都慢落嚟一樣。\n你，依家過得好嗎？"

    print("正在生成粵語獨白音訊...")
    
    # 呼叫 generate 並使用 next() 獲取生成器第一個結果
    result = next(
        model.generate(
            text=text,
            lyrics=lyrics,
            duration=12,  # 設定長度（秒）
            steps=30,     # 推論步數
            seed=7,
        )
    )

    # 輸出音訊維度與取樣率資訊
    print(f"音訊格式: {result.audio.shape}, 取樣率: {result.sample_rate} Hz")

    # 將生成的 numpy array 儲存為 WAV 檔案
    # MiniMax 生成的結果通常是 float32，需確保格式正確
    output_filename = "output/cantonese_monologue.wav"
    # 1. 轉成 NumPy 陣列
    audio_np = np.array(result.audio)

    # 2. 正規化到 -1.0 到 1.0 之間
    if audio_np.size > 0 and np.max(np.abs(audio_np)) > 0:
        audio_np = audio_np / np.max(np.abs(audio_np))

    # 3. 儲存檔案
    wavfile.write(output_filename, result.sample_rate, audio_np)
    print(f"生成成功！檔案已儲存至：{output_filename}")

if __name__ == "__main__":
    main()
