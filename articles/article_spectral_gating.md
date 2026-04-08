---
title: "スペクトルゲーティングをPythonで実装する ― スペクトル減算、Wienerフィルタとの違い"
emoji: "🔊"
type: "tech" # tech or idea
topics: ["python", "dsp", "音声処理", "信号処理"]
published: true
---

## はじめに

スペクトル減算・Wienerフィルタに続く、より実用的なノイズ除去手法が**スペクトルゲーティング（Spectral Gating）**です。

Pythonの人気ノイズ除去ライブラリ [noisereduce](https://github.com/timsainburg/noisereduce) の中核アルゴリズムでもあります。

この記事では、スペクトルゲーティングをスクラッチで実装し、スペクトル減算・Wienerフィルタと定量比較します。

---

## スペクトルゲーティングとは

3手法の考え方の違いを整理します。

| 手法 | アプローチ |
|------|-----------|
| スペクトル減算 | ノイズスペクトルを**引く** |
| Wienerフィルタ | SNRに応じた**比率**で制御 |
| スペクトルゲーティング | ノイズ閾値以下の成分を**ゲートで遮断** |

スペクトルゲーティングのポイントは「閾値」の設計です。ノイズ区間のスペクトルの**平均 + N×標準偏差**を閾値とし、それ以下の成分をゼロに近づけます。

$$threshold = \mu_{noise} + N \cdot \sigma_{noise}$$

$N$（`n_std_thresh`）が大きいほど強くノイズを除去しますが、音声も削れやすくなります。

---

## 必要なライブラリ

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, resample_poly
from scipy.io.wavfile import read, write
from math import gcd
```

Colabで noisereduce を使う場合は事前にインストールしてください。

```
!pip install noisereduce
```

---

## データ準備（LJ Speech + ファンノイズ）

```python
sr_target = 16000

# LJ Speech読み込み・リサンプリング
sr_lj, lj_data = read("LJ001-0001.wav")
lj = resample_poly(
    lj_data.astype(np.float32) / 32767,
    sr_target // gcd(sr_lj, sr_target),
    sr_lj // gcd(sr_lj, sr_target)
)

# fan_noise読み込み・リサンプリング
sr_fan, fan_data = read("fan_noise.wav")
fan = resample_poly(
    fan_data.astype(np.float32) / 32767,
    sr_target // gcd(sr_fan, sr_target),
    sr_fan // gcd(sr_fan, sr_target)
)

sr = sr_target

# 先頭0.5秒の無音区間を追加（ノイズ推定用）
silence = np.zeros(int(sr * 0.5), dtype=np.float32)
clean = np.concatenate([silence, lj])

def add_fan_noise(y, fan, gain=1.0):
    noise = np.tile(fan, (len(y) // len(fan)) + 1)[:len(y)]
    return (y + gain * noise).astype(np.float32)
```

---

## スペクトルゲーティングの実装

```python
def spectral_gating(y, sr, noise_dur=0.5, n_fft=1024, hop=256,
                    n_std_thresh=1.5, prop_decrease=1.0):
    """
    Spectral Gating noise reduction.

    Parameters
    ----------
    y             : input audio signal
    sr            : sample rate
    noise_dur     : duration of noise-only segment at the start [sec]
    n_fft         : FFT size
    hop           : hop length
    n_std_thresh  : threshold = noise_mean + n_std * noise_std
                    (larger = more aggressive gating)
    prop_decrease : how much to suppress gated components
                    (1.0 = zero out, 0.5 = reduce by half)
    """
    _, _, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag, phase = np.abs(Zxx), np.angle(Zxx)

    # ノイズ区間の統計量を計算
    nf = max(1, int(noise_dur * sr / hop))
    noise_mag  = mag[:, :nf]
    noise_mean = np.mean(noise_mag, axis=1, keepdims=True)
    noise_std  = np.std(noise_mag,  axis=1, keepdims=True)

    # 閾値 = mean + n_std_thresh × std
    thresh = noise_mean + n_std_thresh * noise_std

    # ゲート：閾値以上は保持、閾値未満は prop_decrease 分だけ抑圧
    gate_mask = mag >= thresh
    mag_gated = mag * gate_mask + mag * (1 - gate_mask) * (1 - prop_decrease)

    # ISTFT で時間領域に戻す
    _, y_out = istft(mag_gated * np.exp(1j * phase),
                     fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return y_out[:len(y)].astype(np.float32)
```

スペクトル減算・Wienerとの実装の違いはゲートマスクの作り方だけです。

```python
# スペクトル減算：引いてゼロクリップ
mag_out = np.maximum(mag - noise_mean, 0)

# Wienerフィルタ：SNR比率で乗算
H = mag**2 / (mag**2 + noise_mean**2)
mag_out = mag * H

# スペクトルゲーティング：閾値でゲート
gate_mask = mag >= (noise_mean + n_std * noise_std)
mag_out = mag * gate_mask
```

---

## 3手法の定量比較

```python
def spectral_subtraction(y, sr, noise_dur=0.5, n_fft=1024, hop=256):
    _, _, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    nf = max(1, int(noise_dur * sr / hop))
    noise_spec = np.mean(mag[:, :nf], axis=1, keepdims=True)
    _, y_out = istft(np.maximum(mag - noise_spec, 0) * np.exp(1j * phase),
                     fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return y_out[:len(y)].astype(np.float32)

def wiener_filter(y, sr, noise_dur=0.5, n_fft=1024, hop=256):
    _, _, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    nf = max(1, int(noise_dur * sr / hop))
    noise_spec = np.mean(mag[:, :nf], axis=1, keepdims=True)
    H = mag**2 / (mag**2 + noise_spec**2 + 1e-10)
    _, y_out = istft(mag * H * np.exp(1j * phase),
                     fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return y_out[:len(y)].astype(np.float32)

lj_rms  = np.sqrt(np.mean(lj**2))
fan_rms = np.sqrt(np.mean(fan**2))
n = len(clean)

print(f"{'gain':>8} {'SNR(dB)':>8} {'Noisy':>8} {'SS':>8} {'Wiener':>8} {'SG':>8}")
print("-" * 58)

results = []
for gain in [1.0, 5.0, 18.0, 58.0, 183.0]:
    snr   = 20 * np.log10(lj_rms / (fan_rms * gain + 1e-10))
    noisy = add_fan_noise(clean, fan, gain)
    ss    = spectral_subtraction(noisy, sr)
    wi    = wiener_filter(noisy, sr)
    sg    = spectral_gating(noisy, sr)

    r_n  = np.sqrt(np.mean((clean - noisy[:n])**2))
    r_ss = np.sqrt(np.mean((clean - ss[:n])**2))
    r_wi = np.sqrt(np.mean((clean - wi[:n])**2))
    r_sg = np.sqrt(np.mean((clean - sg[:n])**2))
    results.append((snr, r_n, r_ss, r_wi, r_sg))
    print(f"{gain:>8.1f} {snr:>8.1f} {r_n:>8.4f} {r_ss:>8.4f} {r_wi:>8.4f} {r_sg:>8.4f}")
```

実行結果：

```
    gain  SNR(dB)    Noisy       SS   Wiener       SG
----------------------------------------------------------
     1.0     29.7   0.0030   0.0026   0.0027   0.0026
     5.0     15.7   0.0150   0.0104   0.0125   0.0119
    18.0      4.6   0.0541   0.0320   0.0428   0.0397
    58.0     -5.6   0.1743   0.0916   0.1341   0.1218
   183.0    -15.6   0.5498   0.2713   0.4172   0.3723
```

スペクトルゲーティングはWienerフィルタより残差が小さく、スペクトル減算に近い性能を示しています。

---

## n_std_thresh の影響

`n_std_thresh` はスペクトルゲーティングで最も重要なパラメータです。

```python
gain = 18.0  # SNR=4.6dB
noisy = add_fan_noise(clean, fan, gain)

print(f"{'n_std_thresh':>14} {'RMS error':>12}")
print("-" * 28)
for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
    sg = spectral_gating(noisy, sr, n_std_thresh=thresh)
    r  = np.sqrt(np.mean((clean - sg[:n])**2))
    print(f"{thresh:>14.1f} {r:>12.4f}")
```

実行結果：

```
  n_std_thresh    RMS error
           0.5       0.0465
           1.0       0.0431
           1.5       0.0397
           2.0       0.0365
           3.0       0.0312
```

`n_std_thresh` を大きくするほど残差は小さくなりますが、音声成分まで削られ始めます。実用上は1.5〜2.0が目安です。

---

## 可視化

```python
gain  = 18.0
noisy = add_fan_noise(clean, fan, gain)
ss    = spectral_subtraction(noisy, sr)
wi    = wiener_filter(noisy, sr)
sg    = spectral_gating(noisy, sr)
t     = np.arange(n) / sr

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for ax, sig, title, color in zip(axes,
    [clean, noisy, wi, sg],
    ["Clean", "Noisy (SNR=4.6dB)", "Wiener Filter", "Spectral Gating"],
    ["C0", "C1", "C2", "C3"]):
    ax.plot(t[:len(sig)], sig[:len(t)], linewidth=0.4, color=color)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
axes[-1].set_xlabel("Time [s]")
plt.tight_layout()
plt.show()
```

---

## noisereduce との比較

noisereduce ライブラリは内部でスペクトルゲーティングと類似の手法を使っています。

```python
import noisereduce as nr

gain  = 18.0
noisy = add_fan_noise(clean, fan, gain)

# noisereduce は noise_clip（ノイズのみの区間）を渡す
noise_clip = noisy[:int(sr * 0.5)]
denoised_nr = nr.reduce_noise(y=noisy, sr=sr, y_noise=noise_clip)

r_sg = np.sqrt(np.mean((clean - spectral_gating(noisy, sr)[:n])**2))
r_nr = np.sqrt(np.mean((clean - denoised_nr[:n])**2))
print(f"Spectral Gating (scratch): RMS error = {r_sg:.4f}")
print(f"noisereduce:               RMS error = {r_nr:.4f}")
```

---

## 3手法の特性まとめ

| 手法 | 長所 | 短所 | 向いている場面 |
|------|------|------|--------------|
| スペクトル減算 | シンプル・高速 | ミュージカルノイズが出やすい | ノイズ区間が明確な場合 |
| Wienerフィルタ | 音質が自然 | 強いノイズに弱い | バランス重視 |
| スペクトルゲーティング | 効果的・調整しやすい | パラメータ依存 | 実用途の第一選択 |

---

## まとめ

スペクトルゲーティングのポイントは3つです。

1. 閾値（`n_std_thresh`）でゲートのかかり具合を制御できる
2. Wienerフィルタより残差が小さくなることが多い
3. noisereduce ライブラリの中核アルゴリズムでもある

実装はシンプルで、ゲートマスクの計算を追加するだけでスペクトル減算から発展させることができます。

---

基礎からノイズ除去の仕組みを学びたい方はこちら：
👉 [PythonではじめるDSP・音声処理 実践入門](https://zenn.dev/kta805/books/dsp-python-intro)
