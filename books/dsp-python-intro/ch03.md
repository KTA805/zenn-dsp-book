---
title: "スペクトログラムで音を「時間×周波数」で見る"
---


topics: ["python", "dsp", "numpy", "信号処理", "音声処理"]

---

## はじめに

前章のFFTで「音に含まれる周波数」が分かるようになりました。

しかし問題があります。FFTは信号全体を一度に変換するため、「どのタイミングでその周波数が出ていたか」が分かりません。

例えば「ドレミ」を歌ったとき、ドとレとミの音が混ざって見えるだけで、どの順番で出たかが分かりません。

スペクトログラムはこの問題を解決します。短い時間窓でFFTを繰り返すことで、時間とともに周波数がどう変化するかを可視化できます。

---

## 必要なライブラリ

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io.wavfile import write
```

---

## STFTの仕組み

STFT（Short-Time Fourier Transform、短時間フーリエ変換）は、信号を短い窓に分割してFFTを繰り返す処理です。

```python
# 時間とともに周波数が変化する信号を作る
sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 0〜1秒: 440Hz、1〜2秒: 880Hz、2〜3秒: 220Hz
signal = np.zeros(len(t))
signal[t < 1.0] = np.sin(2 * np.pi * 440 * t[t < 1.0])
signal[(t >= 1.0) & (t < 2.0)] = np.sin(2 * np.pi * 880 * t[(t >= 1.0) & (t < 2.0)])
signal[t >= 2.0] = np.sin(2 * np.pi * 220 * t[t >= 2.0])

# 波形を確認
plt.figure(figsize=(10, 3))
plt.plot(t, signal)
plt.xlabel("Time [s]")
plt.title("時間で周波数が変わる信号（440→880→220Hz）")
plt.show()
```

時間波形を見ても、周波数の変化は分かりにくいです。

---

## STFTでスペクトログラムを作る

```python
n_fft = 1024      # FFTサイズ（周波数分解能に影響）
hop = 256         # フレームのずらし幅

f, t_stft, Zxx = stft(signal, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)  # dBスケール

plt.figure(figsize=(12, 5))
plt.pcolormesh(t_stft, f, magnitude_db, shading='gouraud',
               vmin=-60, vmax=0, cmap='viridis')
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.title("スペクトログラム ― 時間×周波数が同時に見える")
plt.colorbar(label="Magnitude [dB]")
plt.ylim(0, 2000)
plt.show()
```

0〜1秒に440Hz、1〜2秒に880Hz、2〜3秒に220Hzの横線が見えます。周波数の変化が時系列で追えるようになりました。

---

## パラメータの意味を理解する

`n_fft` と `hop` の値を変えると、スペクトログラムの見え方が大きく変わります。

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (n_fft, hop, title) in zip(axes, [
    (256, 64,   "小さいFFTサイズ（時間分解能↑、周波数分解能↓）"),
    (4096, 1024, "大きいFFTサイズ（時間分解能↓、周波数分解能↑）"),
]):
    f, t_s, Zxx = stft(signal, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    ax.pcolormesh(t_s, f, mag_db, shading='gouraud',
                  vmin=-60, vmax=0, cmap='viridis')
    ax.set_ylim(0, 2000)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

小さいFFTサイズでは時間方向が細かく見えます。大きいFFTサイズでは周波数が精密に見えます。どちらを優先するかは用途次第です。

---

## hop_lengthの意味

`hop`（hop_length）は見落としやすいパラメータです。

```python
sr = 16000
hop = 512

# 時間分解能の計算
time_resolution = hop / sr * 1000  # ミリ秒

print(f"sr={sr}Hz, hop={hop}")
print(f"時間分解能: {time_resolution:.1f}ms")
print(f"→ {time_resolution:.1f}ms以下の変化は追えない")

# hopを変えると時間分解能が変わる
for h in [128, 256, 512, 1024]:
    res = h / sr * 1000
    print(f"hop={h:4d}: 時間分解能={res:.1f}ms")
```

```
hop= 128: 時間分解能=8.0ms
hop= 256: 時間分解能=16.0ms
hop= 512: 時間分解能=32.0ms
hop=1024: 時間分解能=64.0ms
```

音声認識では通常10〜25ms（hop=160〜400程度）が使われます。

---

## 実際の音声を分析する

声を録音したwavファイルで試してみましょう。手元にwavファイルがない場合は、以下で生成できます。

```python
# テスト用：ピッチが変わる音声を生成
sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# ピッチを徐々に上げる（100→400Hz）
freq = np.linspace(100, 400, len(t))
sweep = np.sin(2 * np.pi * np.cumsum(freq) / sr)
sweep = (sweep * 32767).astype(np.int16)
write("sweep.wav", sr, sweep)
```

```python
from scipy.io.wavfile import read

sr, data = read("sweep.wav")
data = data.astype(np.float32) / 32767

n_fft = 1024
hop = 256
f, t_s, Zxx = stft(data, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
mag_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

plt.figure(figsize=(12, 5))
plt.pcolormesh(t_s, f, mag_db, shading='gouraud',
               vmin=-60, vmax=0, cmap='magma')
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.title("スイープ信号のスペクトログラム ― 周波数が上昇していく様子")
plt.colorbar(label="Magnitude [dB]")
plt.ylim(0, 2000)
plt.show()
```

右肩上がりの明るい線が見えます。時間とともに周波数が上昇しているのが一目で分かります。

---

## スペクトログラムの読み方まとめ

| 見るポイント | 意味 |
|-------------|------|
| 横軸の明るい線 | その時刻に出ていた周波数 |
| 縦方向に並ぶ線 | 倍音構造（声や楽器の特徴） |
| 色の明るさ | 音の強さ（明るいほど強い） |
| 線が動く方向 | ピッチの変化（右肩上がり=音が高くなる） |

---

## まとめ

スペクトログラムは「時間×周波数×強度」を一枚の図に収めた可視化です。

- FFTとの違い：時間変化が見える
- n_fft を大きくすると周波数分解能が上がる（時間分解能は下がる）
- hop を小さくすると時間分解能が上がる（計算量は増える）

次の章では、wavファイルを読み込んで実際の音声を解析します。
