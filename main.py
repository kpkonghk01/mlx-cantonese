import os
import sys
import time

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

DURATION = 230  # 生成長度（秒）
STEPS = 30     # 推論步數（1-30）
SEED = 7

# 風格設定（廣東話獨白、溫柔女聲、無背景音樂）
STYLE_PROMPT = ""

# 歌詞/獨白文本（[Spoken] 標籤控制為說話而非唱歌）
LYRICS = """"""


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

    打的模組不同：ar_one_frame 由 ar.py 內部呼叫 → 打 ar 模組；denoise_chunk 與
    _chunk_starts 由 minimax_music3.py 的 _run_flow 呼叫（denoise_chunk 更是
    `from .euler import` 綁進該命名空間）→ 都要打 minimax_music3 模組（打 euler 無效）。
    """
    import mlx_audio.music.models.minimax_music3.ar as ar_module
    import mlx_audio.music.models.minimax_music3.minimax_music3 as model_module

    total_frames = max(1, int(duration * FRAME_RATE)) + 1

    # 用 nonlocal 閉包共享狀態，避免 stringly-typed 的狀態字典。
    ar_bar = None
    flow_bar = None
    flow_total = None  # Flow 真實段數：從攔截 _chunk_starts 取得

    def make_ar_wrapper(original):
        def wrapped(*args, **kwargs):
            nonlocal ar_bar
            if ar_bar is None:
                ar_bar = tqdm(
                    total=total_frames, desc="① 生成音框(自迴歸)", unit="frame"
                )
            ar_bar.update(1)
            result = original(*args, **kwargs)
            # 模型發出結束訊號(EOS)時內容已生成完，實際格數通常少於估算上限；
            # 把總數校正為當前格數，讓進度條誠實顯示 100% 而非停在中途。
            if getattr(result, "ended", False):
                ar_bar.total = ar_bar.n
                ar_bar.refresh()
            return result

        return wrapped

    def make_chunk_starts_wrapper(original):
        # _run_flow 開始時會呼叫一次 _chunk_starts(frame數) 決定要切幾多段；
        # 攔截它就能在解碼開始前拿到確切段數，給 flow 進度條一個真實總數。
        def wrapped(*args, **kwargs):
            nonlocal flow_total
            starts = original(*args, **kwargs)
            flow_total = len(starts)
            return starts

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
                # flow_total 由攔截 _chunk_starts 取得（真百分比條）；若攔截失敗
                # 而為 None，tqdm 會退回「累計段數 + 速率」顯示，不會出錯。
                flow_bar = tqdm(
                    total=flow_total, desc="② Flow 解碼", unit="chunk"
                )
            flow_bar.update(1)
            return original(*args, **kwargs)

        return wrapped

    _safe_patch(ar_module, "ar_one_frame", make_ar_wrapper)
    _safe_patch(model_module, "_chunk_starts", make_chunk_starts_wrapper)
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


def _format_duration(seconds: float) -> str:
    """把秒數格式化成好讀的 'Hh Mm Ss'（只顯示需要的單位）。"""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _print_summary(result, load_elapsed, gen_elapsed, total_elapsed) -> None:
    """印執行摘要：我們量到的實際牆鐘時間，加上模型自身回報的指標。"""
    print("\n──────── 執行摘要 ────────", flush=True)
    print(f"模型載入耗時：{_format_duration(load_elapsed)}", flush=True)
    print(f"音訊生成耗時：{_format_duration(gen_elapsed)}", flush=True)
    print(f"總計耗時：    {_format_duration(total_elapsed)}", flush=True)

    # 以下來自 GenerationResult；用 getattr 防禦，套件改欄位也不會崩。
    audio_duration = getattr(result, "audio_duration", None)
    if audio_duration:
        print(f"音訊長度：    {audio_duration}", flush=True)
    token_count = getattr(result, "token_count", None)
    if token_count is not None:
        print(f"實際音框數：  {token_count}（自迴歸真正生成的格數）", flush=True)
    rtf = getattr(result, "real_time_factor", None)
    if rtf is not None:
        print(f"實時比 RTF：  {rtf:.2f}×（<1 代表比實時快）", flush=True)
    peak_memory = getattr(result, "peak_memory_usage", None)
    if peak_memory is not None:
        print(f"尖峰記憶體：  {peak_memory:.2f} GB", flush=True)


def main() -> None:
    overall_start = time.perf_counter()

    load_start = time.perf_counter()
    model = _load_model()
    load_elapsed = time.perf_counter() - load_start

    gen_start = time.perf_counter()
    result = _generate_audio(model)
    gen_elapsed = time.perf_counter() - gen_start

    _save_wav(result)
    _print_summary(
        result, load_elapsed, gen_elapsed, time.perf_counter() - overall_start
    )


if __name__ == "__main__":
    main()
