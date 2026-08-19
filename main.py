import os
import sys

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from mlx_audio.music import load

# 強制 stdout / stderr 即時輸出，避免被管線 block-buffer 卡住看不到進度。
# 注意：tqdm 以 \r 寫進度到 stderr，所以 stderr 這行才是進度條即時顯示的關鍵。
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── 設定（集中管理，避免散落的魔術數字）───────────────────────────
MODEL_ID = "mlx-community/MiniMax-Music3-mxfp8"
FRAME_RATE = 25.0  # 與套件 config.py 的 FRAME_RATE 一致，用來估算進度條總格數
OUTPUT_FILENAME = "output/cantonese_monologue.wav"

DURATION = 12  # 生成長度（秒）
STEPS = 30     # 推論步數（1-30）
SEED = 7

# 風格設定（廣東話獨白、溫柔女聲、無背景音樂）
STYLE_PROMPT = "Cantonese monologue, soft female vocal, clear speech, no background music"

# 歌詞/獨白文本（[Spoken] 標籤控制為說話而非唱歌）
LYRICS = (
    "[Spoken]\n"
    "落雨嘅日子，總係會令人諗起好多過去嘅片段。\n"
    "聽住雨聲，好似時間都慢落嚟一樣。\n"
    "你，依家過得好嗎？"
)


def _safe_patch(module, attr, make_wrapper) -> None:
    """把 module.attr 換成 make_wrapper(original) 的結果，並在目標缺失時容錯。

    這是 monkeypatch 的防禦性外殼：若未來的 mlx_audio 版本改名或移除了目標
    函式，這裡只會印警告然後略過進度條，讓生成本身照常進行，而不是拋出裸的
    AttributeError 讓整個程式崩潰。
    """
    original = getattr(module, attr, None)
    if original is None:
        print(
            f"⚠ 進度條略過：找不到 {module.__name__}.{attr}"
            "（mlx_audio 版本可能已變動）；生成仍會正常進行。",
            flush=True,
        )
        return
    setattr(module, attr, make_wrapper(original))


def _attach_progress_bars(duration: float) -> None:
    """替 minimax 生成的兩個階段各掛一條進度條（不改動套件原始碼，只在執行時包裝）。

    Model.generate() 這條路徑完全沒有進度輸出，內部分兩階段：
      ①自迴歸：generate_frame_hiddens() 呼叫 ar_one_frame() 約 duration*FRAME_RATE 次。
      ②Flow 解碼：_run_flow() 逐段呼叫 denoise_chunk() 把音框還原成波形。
    我們在每次呼叫時推進對應的 tqdm，並在階段切換時印提示。

    注意打的模組不同：ar_one_frame 由 ar.py 內部呼叫 → 打 ar 模組；
    denoise_chunk 是 `from .euler import denoise_chunk` 綁進 minimax_music3
    的命名空間後由 _run_flow 呼叫 → 要打 minimax_music3 模組（打 euler 無效）。
    """
    import mlx_audio.music.models.minimax_music3.ar as ar_module
    import mlx_audio.music.models.minimax_music3.minimax_music3 as model_module

    total_frames = max(1, int(duration * FRAME_RATE)) + 1

    # 用 nonlocal 閉包共享兩條進度條，避免 stringly-typed 的狀態字典。
    ar_bar = None
    flow_bar = None

    def make_ar_wrapper(original):
        def wrapped(*args, **kwargs):
            nonlocal ar_bar
            if ar_bar is None:
                ar_bar = tqdm(
                    total=total_frames, desc="① 生成音框(自迴歸)", unit="frame"
                )
            ar_bar.update(1)
            return original(*args, **kwargs)

        return wrapped

    def make_flow_wrapper(original):
        def wrapped(*args, **kwargs):
            nonlocal flow_bar
            # 第一次被呼叫 = 剛從自迴歸切到 flow 階段：收尾①、印提示、開②。
            if flow_bar is None:
                if ar_bar is not None:
                    ar_bar.close()
                print(
                    "\n自迴歸完成，進入 Flow 解碼階段（把音框還原成波形）...",
                    flush=True,
                )
                # chunk 總數依實際生成長度浮動，難以事前精算，故用 total=None：
                # tqdm 會顯示「已完成段數 + 速率」而非百分比。
                flow_bar = tqdm(desc="② Flow 解碼", unit="chunk")
            flow_bar.update(1)
            return original(*args, **kwargs)

        return wrapped

    _safe_patch(ar_module, "ar_one_frame", make_ar_wrapper)
    _safe_patch(model_module, "denoise_chunk", make_flow_wrapper)


def _load_model():
    """載入模型；失敗時給出友善訊息並以非零碼結束。"""
    print(f"正在載入模型: {MODEL_ID}...", flush=True)
    print("（首次執行會從 HuggingFace 下載數 GB 模型，請耐心等候）", flush=True)
    try:
        return load(MODEL_ID)
    except Exception as exc:  # 下載中斷、模型 ID 錯誤、磁碟不足等
        raise SystemExit(f"✗ 模型載入失敗：{exc}")


def _generate_audio(model):
    """跑生成並取回第一個結果；失敗時給出友善訊息並以非零碼結束。"""
    print("正在生成粵語獨白音訊...", flush=True)
    _attach_progress_bars(DURATION)
    try:
        return next(
            model.generate(
                text=STYLE_PROMPT,
                lyrics=LYRICS,
                duration=DURATION,
                steps=STEPS,
                seed=SEED,
            )
        )
    except StopIteration:
        raise SystemExit("✗ 生成器沒有產生任何音訊結果。")
    except Exception as exc:  # 生成過程中的模型/數值錯誤
        raise SystemExit(f"✗ 音訊生成失敗：{exc}")


def _save_wav(result) -> None:
    """把生成結果正規化後存成 WAV。"""
    print(
        f"\n音訊格式: {result.audio.shape}, 取樣率: {result.sample_rate} Hz",
        flush=True,
    )

    # 確保輸出目錄存在，否則 wavfile.write 會在最後一步崩潰。
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    audio_np = np.array(result.audio)

    # 正規化到 -1.0 ~ 1.0
    if audio_np.size > 0 and np.max(np.abs(audio_np)) > 0:
        audio_np = audio_np / np.max(np.abs(audio_np))

    try:
        wavfile.write(OUTPUT_FILENAME, result.sample_rate, audio_np)
    except OSError as exc:  # 權限、磁碟空間等寫檔問題
        raise SystemExit(f"✗ 儲存檔案失敗：{exc}")
    print(f"生成成功！檔案已儲存至：{OUTPUT_FILENAME}", flush=True)


def main() -> None:
    model = _load_model()
    result = _generate_audio(model)
    _save_wav(result)


if __name__ == "__main__":
    main()
