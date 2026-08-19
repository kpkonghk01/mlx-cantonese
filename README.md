# mlx-cantonese

用 [MiniMax-Music3](https://huggingface.co/mlx-community/MiniMax-Music3-mxfp8) 喺
Apple Silicon（MLX）上生成**粵語獨白**音訊。

> 呢份 README 假設你唔熟 Python。照住抄、逐步貼指令就得。

---

## 一次性設定（只需做一次）

### 1. 裝 uv（Python 的套件/環境管理工具）

我哋用 `uv` 幫你自動搞掂 Python 版本同所有依賴，你唔使識 Python 內部運作。
喺 Terminal（終端機）貼呢句：

```bash
brew install uv
```

> 已經裝過就會話你 already installed，唔會出事，可以直接落下一步。
> 冇 Homebrew（`brew`）的話，去 https://brew.sh 跟指示裝，或者用
> `curl -LsSf https://astral.sh/uv/install.sh | sh` 裝 uv。

### 2. 入到專案資料夾

```bash
cd ~/Documents/workspace/mlx-cantonese
```

> `cd` = 「入去邊個資料夾」。之後所有指令都要喺呢個資料夾裡面行。

### 3. 安裝依賴

```bash
uv sync
```

呢句會**自動**建立一個獨立環境、裝好 `mlx-audio`、`numpy`、`scipy`、`tqdm`
（全部喺 `pyproject.toml` 列明），版本鎖定喺 `uv.lock`。做完就準備好。

---

## 執行（每次想生成音訊就做呢步）

```bash
uv run python -u main.py
```

拆解俾你明：
- `uv run` = 用啱嘅環境去行（唔使自己手動 activate 乜嘢）。
- `python -u` = **關掉輸出緩衝**，令進度即時顯示。`-u` 好緊要，冇佢可能睇唔到進度。
- `main.py` = 主程式。

### 你會見到咩（代表正常運作）

程式**唔會**即刻有反應係正常嘅，順序係咁：

1. `正在載入模型...`
   → **第一次執行會由網上下載數 GB 模型**，可能等幾分鐘，靜靜地下載，唔係當機。
   （之後再行就會用本機快取，快好多。）
2. `① 生成音框(自迴歸)` 進度條 —— 一格一格行。
3. `自迴歸完成，進入 Flow 解碼階段...` 之後出 `② Flow 解碼` 進度條。
4. `生成成功！檔案已儲存至：output/cantonese_monologue.wav`

生成好嘅音訊喺 **`output/cantonese_monologue.wav`**，用任何播放器打開就聽到。

---

## 想改內容？

打開 `main.py`，最上面呢幾個設定就係你會想調嘅嘢：

| 設定 | 意思 |
|------|------|
| `DURATION` | 生成長度（秒），例如 `12` |
| `STEPS` | 推論步數（1–30），越大通常越精細但越慢 |
| `SEED` | 亂數種子；同一個 seed 出同樣結果，改咗就換個變化 |
| `STYLE_PROMPT` | 風格描述（英文），例如聲線、有冇背景音樂 |
| `LYRICS` | 要講嘅內容；`[Spoken]` 標籤代表「講嘢」而唔係「唱歌」 |

改完存檔，再行一次 `uv run python -u main.py` 就得。

---

## 常見問題

- **成個畫面靜咗好耐冇反應？**
  第一次執行 = 大機率喺下載模型（數 GB）。畀啲耐性，唔好中途取消。
- **完全冇任何文字輸出？**
  確認你有加 `-u`（即 `uv run python -u main.py`）。
- **`command not found: uv`？**
  你未裝 uv，返去上面「一次性設定」第 1 步。
- **`command not found: brew`？**
  你未裝 Homebrew，去 https://brew.sh 跟指示裝。
