---
title: "ピッチ（F0）を推定する"
free: false
---


topics: ["python", "dsp", "信号処理", "librosa", "音声処理"]

---

## はじめに

音には「高さ」があります。ソプラノの声は高く、バスの声は低い。この「音の高さ」を物理的に表したものが基本周波数（F0: Fundamental frequency）です。

ピッチ検出はF0を時間ごとに推定する処理です。以下のような用途で使われます。

- 音声認識の前処理（有声音・無声音の判定）
- 音楽解析（音程の検出）
- 話者分析（声の特徴の把握）
- 歌声の音程補正（ピッチシフター）

---

## 必要なライブラリ

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import librosa
```

---

## サンプル音声の準備

```python
# 自分の声で試す場合はこのコードは不要
# 以下はテスト用の音声生成コード

sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# 時間とともにピッチが変わる音声（100Hz → 300Hz）
# 実際の声のようにピッチが滑らかに変化する
f0_trajectory = np.linspace(100, 300, len(t))
phase = 2 * np.pi * np.cumsum(f0_trajectory) / sr

signal = (np.sin(phase) +
          0.5 * np.sin(2 * phase) +
          0.3 * np.sin(3 * phase))
signal = (signal / np.max(np.abs(signal))).astype(np.float32)

# 途中に無声区間を入れる（子音のような区間）
signal[int(1.0*sr):int(1.3*sr)] = 0.0

write("pitch_test.wav", sr, (signal * 32767).astype(np.int16))
print("pitch_test.wav を生成しました")
print(f"音声の長さ: {duration}秒")
print(f"ピッチ範囲: 100Hz → 300Hz（途中1.0〜1.3秒は無声区間）")
```

---

## pyin によるピッチ推定

```python
# 音声を読み込む
y, sr = librosa.load("pitch_test.wav", sr=None)

# pyin でピッチを推定
# fmin, fmax で探索する周波数の範囲を指定する
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),  # 約65Hz
    fmax=librosa.note_to_hz("C7"),  # 約2093Hz
    sr=sr
)

times = librosa.times_like(f0, sr=sr)

print(f"フレーム数: {len(f0)}")
print(f"有声フレーム数: {voiced_flag.sum()}")
print(f"無声フレーム数: {(~voiced_flag).sum()}")
print(f"検出されたF0の範囲: {np.nanmin(f0):.0f}Hz 〜 {np.nanmax(f0):.0f}Hz")
```

---

## 結果の可視化

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# 波形
axes[0].plot(np.arange(len(y)) / sr, y, linewidth=0.5)
axes[0].set_ylabel("Amplitude")
axes[0].set_title("波形")

# F0の推定値
axes[1].plot(times, f0, color='C0', linewidth=1.5, label='F0')
axes[1].set_ylabel("Frequency [Hz]")
axes[1].set_title("推定されたF0（NaNは無声区間）")
axes[1].legend()
axes[1].set_ylim(0, 500)

# 有声・無声の確率
axes[2].plot(times, voiced_probs, color='C2', linewidth=1.0)
axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='threshold=0.5')
axes[2].set_ylabel("Voiced probability")
axes[2].set_xlabel("Time [s]")
axes[2].set_title("有声音の確率")
axes[2].legend()
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

F0のグラフで、値がある区間が有声音（母音など）、NaNになっている区間が無声音（子音・無音）です。

---

## 自分の声でピッチを分析する

スマートフォンやマイクで録音して試してみてください。

```python
# 自分の声を録音したwavファイルで分析する
sr, data = read("myvoice.wav")  # 自分の声のファイル名に変更
if data.ndim == 2:
    data = data[:, 0]
y = data.astype(np.float32) / 32767

# 人の声に適したパラメータ
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=70,    # 男性の低い声〜
    fmax=500,   # 〜女性の高い声
    sr=sr
)

times = librosa.times_like(f0, sr=sr)
valid_f0 = f0[voiced_flag]

if len(valid_f0) > 0:
    print(f"平均ピッチ: {np.mean(valid_f0):.0f}Hz")
    print(f"最低ピッチ: {np.min(valid_f0):.0f}Hz")
    print(f"最高ピッチ: {np.max(valid_f0):.0f}Hz")
    print()
    print("参考値：")
    print("  男性の話し声: 80〜180Hz")
    print("  女性の話し声: 150〜300Hz")
```

---

## ピッチヒストグラムで声の特徴を見る

```python
valid_f0 = f0[voiced_flag]

if len(valid_f0) > 0:
    plt.figure(figsize=(10, 4))
    plt.hist(valid_f0, bins=50, edgecolor='white', linewidth=0.5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Count")
    plt.title("ピッチの分布（有声区間のみ）")
    plt.axvline(x=np.mean(valid_f0), color='red', linestyle='--',
                label=f'平均: {np.mean(valid_f0):.0f}Hz')
    plt.legend()
    plt.show()
```

---

## ピッチと音符の対応

音楽の音符とピッチの対応も確認できます。

```python
# ピッチを音符名に変換
def hz_to_note_name(hz):
    if hz <= 0 or np.isnan(hz):
        return "---"
    note_number = librosa.hz_to_midi(hz)
    note_name = librosa.midi_to_note(int(round(note_number)))
    return note_name

# 代表的な音符のピッチ
representative_pitches = {
    "ド（C4）": 261.6,
    "レ（D4）": 293.7,
    "ミ（E4）": 329.6,
    "ファ（F4）": 349.2,
    "ソ（G4）": 392.0,
    "ラ（A4）": 440.0,
    "シ（B4）": 493.9,
}

print("音符とピッチの対応:")
for name, freq in representative_pitches.items():
    print(f"  {name}: {freq:.1f}Hz")
```

---

## ピッチ検出の精度に影響する要因

```python
# fmin/fmax の設定が重要
print("=== fmin/fmax の設定例 ===")
for fmin, fmax, desc in [
    (80,  300, "男性の話し声"),
    (150, 400, "女性の話し声"),
    (50,  500, "広い範囲（歌声含む）"),
    (200, 2000,"笛・フルートなど"),
]:
    print(f"  fmin={fmin:4d}Hz, fmax={fmax:4d}Hz → {desc}")
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| 関数 | `librosa.pyin` |
| 出力1 | F0の時系列（NaN=無声） |
| 出力2 | voiced_flag（有声/無声の判定） |
| 出力3 | voiced_probs（有声確率） |
| 重要パラメータ | fmin, fmax（探索する周波数範囲） |

---

## シリーズ一覧

1. サンプリング定理とエイリアシング
2. FFTで周波数スペクトルを見る
3. スペクトログラムで音を可視化する
4. wavファイルを読み込んで解析する
5. メルスペクトログラムを作る
6. MFCCで音声を数値化する
7. 音声分類モデルを作る
8. スペクトル減算でノイズを除去する
9. ピッチ（F0）を推定する ← 今回
10. ノイズ除去の効果を実験で検証する
