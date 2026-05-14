---
title: "WhisperにHPFを入れたら日本語音声認識精度が改善した"
emoji: "🎤"
type: "tech"
topics:
  - whisper
  - python
  - 音声認識
  - ai
  - dsp
published: false
---

# はじめに

Whisperを使って日本語音声認識を試していたところ、ノイズ環境で認識精度がかなり悪化しました。

そこで今回は、以下の前処理を比較し、CER（Character Error Rate）を評価してみました。

- Normalization
- HPF（High Pass Filter）
- VAD（Voice Activity Detection）
- fixed chunk
- overlap chunk

結論から言うと、今回最も効果があったのは100Hz HPFでした。

一方、VADやchunkingは今回の条件では逆に性能悪化となりました。

「効かなかった手法」も含めて面白い結果が得られたので、記録としてまとめます。

---

# 実験環境

- Python 3.13
- openai-whisper
- librosa
- scipy
- jiwer
- silero-vad

Whisper modelは `small` を使用しました。

```python
model = whisper.load_model("small")
```

---

# 評価指標

評価にはCER（Character Error Rate）を使用しました。

```python
from jiwer import cer

error = cer(reference, hypothesis)
```

---

# 実験対象

以下2種類の音声を使用しました。

- clean音声
- noisy音声（0dBノイズ付加）

---

# Baseline

まずはそのままWhisperへ入力。

| Method | CER |
|---|---|
| Clean | 0.060 |
| Noisy | 0.148 |

ノイズによってCERが大きく悪化しました。

---

# Normalization

まずは単純なNormalizationを試しました。

```python
y = y / np.max(np.abs(y))
```

しかし結果はほぼ変化なし。

| Method | CER |
|---|---|
| Normalization | 0.148 |

単純な音量差ではなく、ノイズ帯域そのものが問題だと考えました。

---

# FFT確認

FFTを確認すると、100Hz以下に強い低周波成分が存在していました。

そこで、100Hz HPFを適用しました。

---

# HPF

## 実装

```python
from scipy.signal import butter, filtfilt

def highpass_filter(y, sr, cutoff=100):

    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        N=4,
        Wn=normal_cutoff,
        btype='high'
    )

    filtered = filtfilt(b, a, y)

    return filtered
```

---

# HPF結果

| Method | CER |
|---|---|
| HPF (100Hz) | 0.114 |

CER改善を確認しました。

---

# cutoff sweep

cutoff周波数も比較してみました。

| cutoff | CER |
|---|---|
| 50Hz | 0.148 |
| 100Hz | 0.114 |
| 200Hz | 0.182 |
| 300Hz | 0.175 |

100Hz付近が最も良い結果となりました。

200Hz以上では音声成分まで削ってしまい、逆に性能悪化したと考えられます。

---

# Spectrogram確認

Spectrogramも確認しました。

FFTでは100Hz以下の成分減衰が確認できましたが、Spectrogramでは見た目の差はそこまで大きくありませんでした。

しかしCER改善が確認できたため、

- 人間には分かりにくい低周波ノイズ
- Whisper内部特徴量への影響

が存在していた可能性があります。

---

# VAD

次にSilero VADを試しました。

```python
from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)
```

---

# VAD結果

| Method | CER |
|---|---|
| VAD | 0.304 |

大きく悪化しました。

---

# なぜ悪化したのか？

今回の結果から、Whisperはかなり長文脈依存型モデルであることが見えてきました。

VADによって発話区間を細かく切り出した結果、

- 文脈情報の断裂
- 単語途中切断
- segment分断

が発生し、認識性能が悪化した可能性があります。

---

# fixed chunk

固定長chunkも試しました。

- 5秒chunk
- 10秒chunk

などを比較。

しかし今回の条件では改善せず。

| Method | CER |
|---|---|
| Fixed Chunk | 0.358 |

---

# overlap chunk

次にoverlap chunkも試しました。

例えば：

- 0-10秒
- 8-18秒
- 16-26秒

のように、2秒重複を持たせました。

---

# overlap結果

| Method | CER |
|---|---|
| Overlap Chunk | 0.540 |

さらに悪化。

原因は、chunk間の重複文をそのまま連結したためでした。

---

# Whisper標準segmentについて

今回かなり面白かったのは、

「Whisper自身が内部でsegment処理を持っている」

という点です。

つまり、

```python
result = model.transcribe("audio.wav")
```

を実行した時点で、Whisper内部では既にsegment分割が行われています。

```python
print(result["segments"])
```

で確認可能です。

今回の結果から、

- Whisper内部segment
- 外部chunking

が競合し、性能悪化を招いた可能性があると考えています。

---

# 最終結果

| Method | CER |
|---|---|
| Clean | 0.060 |
| Noisy | 0.148 |
| Normalization | 0.148 |
| HPF (100Hz) | 0.114 |
| VAD | 0.304 |
| HPF + VAD | 0.283 |
| Fixed Chunk | 0.358 |
| Overlap Chunk | 0.540 |

---

# まとめ

今回の実験では、以下が確認できました。

- 単純Normalizationは効果なし
- 低周波ノイズに対してHPFが有効
- Whisperは長文脈依存が強い
- 雑なchunkingは性能悪化する
- overlapには後処理が必要

特に、

「効かなかった手法」

も含めて確認できたのは面白かったです。

---
