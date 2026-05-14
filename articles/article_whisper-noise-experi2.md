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