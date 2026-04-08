# Noise Reduction vs. ASR Accuracy ― Whisper + WER Comparison

topics: ["python", "dsp", "音声処理", "whisper", "音声認識"]

---

## はじめに

スペクトル減算・Wienerフィルタ・スペクトルゲーティングを比較するとき、RMS誤差だけでは不十分です。

RMS誤差が小さい＝音質が良い、とは限らないからです。スペクトル減算は音声成分ごと削ることでRMS誤差を小さくしますが、その分だけ音声認識の精度が落ちる可能性があります。

この記事では、**OpenAI Whisper で実際に文字起こしをして WER（単語誤り率）で比較**します。

---

## WER（Word Error Rate）とは

音声認識の標準的な評価指標です。

$$WER = \frac{S + D + I}{N}$$

- $S$：置換（Substitution）：違う単語に変換された数
- $D$：削除（Deletion）：抜けた単語の数
- $I$：挿入（Insertion）：余計に挿入された単語の数
- $N$：正解テキストの総単語数

WERが低いほど認識精度が高いです。0.0が完璧、1.0が最悪です。

---

## 実行環境について

**Google Colaboratoryでの実行を推奨します。**

WhisperはPyTorchに依存しており、ローカル環境（Windows + miniconda等）ではDLLのバージョン不一致などで以下のようなエラーが発生することがあります。

```
OSError: [WinError 127] 指定されたプロシージャが見つかりません。
Error loading "torch/lib/shm.dll" or one of its dependencies.
```

Colabであれば環境構築不要でGPUも使えるため、数分で実行できます。

---

## Colabのセットアップ

ブラウザで https://colab.research.google.com を開き、新しいノートブックを作成してください。

```python
!pip install openai-whisper jiwer noisereduce scipy
```

---

## 必要なライブラリ

```python
import numpy as np
import whisper
import soundfile as sf
from jiwer import wer
from scipy.signal import stft, istft, resample_poly
from scipy.io.wavfile import read, write
from math import gcd
import tempfile
import os
```

---

## データ準備

LJ001-0001.wav と fan_noise.wav を Google Drive に置いてマウントするか、直接アップロードしてください。

```python
from google.colab import files

# ファイルをアップロードする場合
uploaded = files.upload()  # LJ001-0001.wav と fan_noise.wav をアップロード
```

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

# LJ001-0001の正解テキスト
REFERENCE = (
    "printing in the only sense with which we are at present concerned "
    "differs from most if not from all the arts and crafts represented "
    "in the exhibition"
)

print(f"Clean audio: {len(clean)/sr:.2f}sec")
print(f"Reference  : {REFERENCE}")
```

---

## ノイズ除去関数

```python
def add_fan_noise(y, fan, gain=1.0):
    noise = np.tile(fan, (len(y) // len(fan)) + 1)[:len(y)]
    return (y + gain * noise).astype(np.float32)

def spectral_subtraction(y, sr, noise_dur=0.5, n_fft=1024, hop=256):
    _, _, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    nf = max(1, int(noise_dur * sr / hop))
    noise_spec = np.mean(mag[:, :nf], axis=1, keepdims=True)
    mag_d = np.maximum(mag - noise_spec, 0)
    _, y_out = istft(mag_d * np.exp(1j * phase),
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

def spectral_gating(y, sr, noise_dur=0.5, n_fft=1024, hop=256,
                    n_std_thresh=1.5, prop_decrease=1.0):
    _, _, Zxx = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    nf = max(1, int(noise_dur * sr / hop))
    noise_mag  = mag[:, :nf]
    noise_mean = np.mean(noise_mag, axis=1, keepdims=True)
    noise_std  = np.std(noise_mag,  axis=1, keepdims=True)
    thresh     = noise_mean + n_std_thresh * noise_std
    gate_mask  = mag >= thresh
    mag_gated  = mag * gate_mask + mag * (1 - gate_mask) * (1 - prop_decrease)
    _, y_out = istft(mag_gated * np.exp(1j * phase),
                     fs=sr, nperseg=n_fft, noverlap=n_fft-hop)
    return y_out[:len(y)].astype(np.float32)

def noisereduce_denoise(y, sr, noise_dur=0.5):
    import noisereduce as nr
    noise_clip = y[:int(sr * noise_dur)]
    return nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip).astype(np.float32)
```

---

## Whisperで文字起こし → WER計算

```python
# Whisperモデルのロード（base で十分）
model = whisper.load_model("base")

def transcribe_and_wer(audio, sr, reference, model):
    """音声を一時ファイルに保存してWhisperで文字起こし、WERを計算する"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmpfile = f.name
    write(tmpfile, sr, (audio * 32767).astype(np.int16))
    result = model.transcribe(tmpfile, language="en")
    hypothesis = result["text"].strip().lower()
    # 句読点を除去してWERを計算
    import re
    hypothesis = re.sub(r'[^\w\s]', '', hypothesis)
    score = wer(reference, hypothesis)
    os.unlink(tmpfile)
    return hypothesis, score
```

---

## SNRを変えて全手法のWERを比較

```python
lj_rms  = np.sqrt(np.mean(lj**2))
fan_rms = np.sqrt(np.mean(fan**2))

print(f"{'SNR(dB)':>8} {'Clean':>8} {'Noisy':>8} {'SS':>8} {'Wiener':>8} {'SG':>8} {'NR':>8}")
print("-" * 64)

all_results = []
for gain in [1.0, 5.0, 18.0, 58.0]:
    snr   = 20 * np.log10(lj_rms / (fan_rms * gain + 1e-10))
    noisy = add_fan_noise(clean, fan, gain)

    # 各手法で除去
    ss = spectral_subtraction(noisy, sr)
    wi = wiener_filter(noisy, sr)
    sg = spectral_gating(noisy, sr)
    nr = noisereduce_denoise(noisy, sr)

    # Whisperで文字起こし
    _, wer_clean = transcribe_and_wer(clean, sr, REFERENCE, model)
    _, wer_noisy = transcribe_and_wer(noisy, sr, REFERENCE, model)
    _, wer_ss    = transcribe_and_wer(ss,    sr, REFERENCE, model)
    _, wer_wi    = transcribe_and_wer(wi,    sr, REFERENCE, model)
    _, wer_sg    = transcribe_and_wer(sg,    sr, REFERENCE, model)
    _, wer_nr    = transcribe_and_wer(nr,    sr, REFERENCE, model)

    all_results.append((snr, wer_clean, wer_noisy, wer_ss, wer_wi, wer_sg, wer_nr))
    print(f"{snr:>8.1f} {wer_clean:>8.3f} {wer_noisy:>8.3f} "
          f"{wer_ss:>8.3f} {wer_wi:>8.3f} {wer_sg:>8.3f} {wer_nr:>8.3f}")
```

実行結果（Whisper base, CPU）：

```
 SNR(dB)    Clean    Noisy       SS   Wiener       SG       NR
----------------------------------------------------------------
    29.7    0.000    0.000    0.000    0.000    0.000    0.000
    15.7    0.000    0.000    0.000    0.000    0.000    0.000
     4.6    0.000    0.000    0.000    0.000    0.000    0.185
    -5.6    0.000    0.889    0.667    0.667    0.815    0.926
```

`FP16 is not supported on CPU` という警告が出ることがありますが、CPUで実行している場合の正常な動作です。結果には影響しません。

---

## 結果の可視化

```python
import matplotlib.pyplot as plt

snrs     = [r[0] for r in all_results]
labels   = ['Clean', 'Noisy', 'SS', 'Wiener', 'SG', 'noisereduce']
colors   = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
markers  = ['o', 's', '^', 'D', 'v', 'p']

plt.figure(figsize=(10, 5))
for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
    wers = [r[i+1] for r in all_results]
    plt.plot(snrs, wers, marker=marker, linestyle='-',
             label=label, color=color)

plt.xlabel("SNR [dB]")
plt.ylabel("WER (lower is better)")
plt.title("SNR vs. WER: Noise Reduction Methods Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.gca().invert_xaxis()  # SNR高い方を左に
plt.tight_layout()
plt.show()
```

---

## RMS誤差とWERの比較

```python
# RMS誤差とWERが一致しない例を確認
gain  = 18.0  # SNR=4.6dB
noisy = add_fan_noise(clean, fan, gain)
n     = len(clean)

ss = spectral_subtraction(noisy, sr)
wi = wiener_filter(noisy, sr)
sg = spectral_gating(noisy, sr)

methods = {"SS": ss, "Wiener": wi, "SG": sg}

print(f"{'Method':>10} {'RMS error':>12} {'WER':>8}")
print("-" * 32)
for name, y_den in methods.items():
    rms  = np.sqrt(np.mean((clean - y_den[:n])**2))
    _, w = transcribe_and_wer(y_den, sr, REFERENCE, model)
    print(f"{name:>10} {rms:>12.4f} {w:>8.3f}")
```

実行結果（SNR=4.6dB）：

```
    Method    RMS error      WER
--------------------------------
        SS       0.0320    0.000
    Wiener       0.0428    0.000
        SG       0.0397    0.000
```

RMS誤差はSS < SG < Wienerの順ですが、WERはすべて0.000で差がありません。RMS誤差が最も小さいSSが「最も良い手法」とは言えないことが分かります。このSNR条件ではどの手法も音声認識精度の観点からは同等です。

---

## 考察

結果から読み取れることを整理します。

**SNR=29.7〜15.7dB（ノイズが小さい）**

全手法でWER=0.000です。Whisper baseはこの程度のノイズなら問題なく認識できます。ノイズ除去をしなくても精度は落ちません。

**SNR=4.6dB（ノイズと音声が同程度）**

自前実装の4手法（Clean / Noisy / SS / Wiener / SG）はすべてWER=0.000を維持しました。一方、**noisereduceだけWER=0.185に悪化**しています。

これは意外な結果です。高機能なライブラリが自前実装より悪い結果になっています。noisereduceはデフォルトのパラメータが積極的なノイズ除去に設定されており、このSNR条件では音声成分まで削りすぎた可能性があります。

またRMS誤差とWERを並べると、SS（RMS=0.0320）・SG（RMS=0.0397）・Wiener（RMS=0.0428）の順で残差が違うにもかかわらず、WERはすべて0.000で差がありません。**RMS誤差が小さい＝音声認識精度が高いとは言えない**ことがここで確認できます。

**SNR=-5.6dB（ノイズの方が大きい）**

ここで手法間の差が明確に出ました。

| 手法 | WER | 解釈 |
|------|-----|------|
| Clean | 0.000 | ノイズなしは完璧 |
| Noisy | 0.889 | ほぼ認識不能 |
| SS | 0.667 | Noisyより改善 |
| Wiener | 0.667 | SSと同等 |
| SG | 0.815 | SS・Wienerより悪い |
| noisereduce | 0.926 | Noisyより悪化 |

SSとWienerが最も良い結果になりました。スペクトルゲーティングはこのSNR条件では積極的にゲートをかけすぎて音声も削られた可能性があります。noisereduceは逆効果でした。

**重要な発見：RMS誤差とWERは一致しない**

前の記事でRMS誤差を比較したとき、SSが最も小さい値でした。しかしWERではSS・Wiener・SGはほぼ同等です。RMS誤差だけで「SSが最も良い」と結論づけるのは誤りでした。

---

## まとめ

- SNR≧15dBではWhisperはノイズ除去なしでも正確に認識できる
- SNR=4.6dB付近では自前実装の手法はすべて有効、noisereduceのデフォルト設定は要注意
- SNR=-5.6dBの極端な条件ではSSとWienerがWER改善に最も効果的
- RMS誤差が小さい手法が必ずしもWERが低いわけではない
- 音声認識の前処理としてノイズ除去を評価するにはWERが適切な指標

---

基礎からノイズ除去の仕組みを学びたい方はこちら：
👉 [PythonではじめるDSP・音声処理 実践入門](https://zenn.dev/kta805/books/dsp-python-intro)
