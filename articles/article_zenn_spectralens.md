---
title: "ブラウザだけで完結する音声スペクトル解析ツールをJSで作った話 — FFT・メルスペクトログラム・MFCCをゼロ実装"
emoji: "🎙"
type: "tech"
topics: ["python", "dsp", "audio", "javascript", "mfcc"]
published: false
---

# ブラウザだけで完結する音声スペクトル解析ツールをJSで作った話 — FFT・メルスペクトログラム・MFCCをゼロ実装

## はじめに

音声処理の勉強をしていると、「手元の音声ファイルをすぐ分析したい」という場面が頻繁にあります。

librosaをインポートしてJupyterで可視化する、という手順は慣れれば簡単ですが、毎回環境を立ち上げるのが少し面倒です。また「Pythonを書かない人に音声データを確認してもらいたい」というケースでは、そもそも環境を用意してもらうところからが大変になります。

そこで**ブラウザにファイルをドロップするだけで、FFT・メルスペクトログラム・MFCCを即座に可視化できるツール**を作りました。

🔗 **[SpectraLens — Live Demo](https://YOUR_USERNAME.github.io/spectralens/)**

サーバーへの音声データ送信はなく、すべての処理はブラウザ内で完結する仕様にしました。
プライバシー、セキュリティ面でも安心して使える設計になっています。
興味がある方は是非使ってみていただきたいです。感想お待ちしています。

---

## 何ができるか

3つのタブで異なる視点から音声を分析できます。

**Classicタブ**：波形・FFTスペクトル・スペクトログラム・周波数帯域別エネルギーを表示します。音声の基本的な特性確認に使えます。

**Mel / MFCCタブ**：メルスペクトログラム（40メルビン）・MFCCヒートマップ（13次元×時間）・MFCC平均係数バーチャートを表示します。音声認識・分類の前処理確認に使えます。

**Δ Delta MFCCタブ**：1次デルタMFCC（特徴量の速度）・2次デルタMFCC（加速度）・係数別時系列チャートを表示します。音声の動的特徴の確認に使えます。

統計値としてピーク周波数・RMSレベル・サンプルレート・再生時間・スペクトル重心もリアルタイムで表示されます。

---

## なぜPythonではなくJavaScriptで実装したか

最初はPythonバックエンド（FastAPI + librosa）で作ることを考えました。しかし次の理由でブラウザ完結型を選びました。

**サーバー費用がゼロになります。** GitHub Pagesに静的ファイルを置くだけで公開できます。FastAPIをホスティングすると月数百〜数千円のコストが継続的にかかります。

**音声データがサーバーに送信されません。** 会議の録音や研究データなど、外部に送りたくないファイルでも使えます。これはセキュリティ・プライバシーの観点を考慮しています。

**インストール不要で誰でも使えます。** URLを送るだけで相手がすぐ試せます。

---

## 技術的な実装の話

### Web Audio APIで音声を読み込む

ブラウザにはWeb Audio APIという強力なAPIが標準搭載されています。WAV・MP3・OGGなどの音声ファイルをデコードして、Float32Arrayとして波形データを取り出せます。

```javascript
const audioCtx = new AudioContext();
const arrayBuffer = await file.arrayBuffer();
const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
const data = audioBuffer.getChannelData(0); // モノラル化
```

librosaの`librosa.load()`に相当する処理がこれだけで書けます。

### FFTをJavaScriptでゼロ実装した

最初はWeb Audio APIの`AnalyserNode`を使うことを考えました。しかしこれはリアルタイム処理向けで、ファイル全体のFFTには向いていません。そこでRadix-2のCooley-Tukey FFTアルゴリズムをJavaScriptでゼロから実装しました。

```javascript
function fft(re, im) {
  const N = re.length;
  if (N <= 1) return;

  // 偶数・奇数インデックスに分割（分割統治）
  const eRe = re.filter((_, i) => i % 2 === 0);
  const eIm = im.filter((_, i) => i % 2 === 0);
  const oRe = re.filter((_, i) => i % 2 === 1);
  const oIm = im.filter((_, i) => i % 2 === 1);

  fft(eRe, eIm);
  fft(oRe, oIm);

  for (let k = 0; k < N / 2; k++) {
    const angle = -2 * Math.PI * k / N;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    const tRe = cos * oRe[k] - sin * oIm[k];
    const tIm = sin * oRe[k] + cos * oIm[k];
    re[k]       = eRe[k] + tRe;
    im[k]       = eIm[k] + tIm;
    re[k + N/2] = eRe[k] - tRe;
    im[k + N/2] = eIm[k] - tIm;
  }
}
```

Hanning窓を適用してからFFTを実行し、パワースペクトルをdBスケールに変換しています。

```javascript
// Hanning窓
for (let i = 0; i < N; i++) {
  windowed[i] = data[i] * 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
}

// dBへの変換
const dB = 20 * Math.log10(magnitude / N + 1e-9) + 96;
```

### メルフィルタバンクをゼロ実装した

MFCCの計算で一番実装が面倒なのがメルフィルタバンクです。周波数軸をメルスケールに変換してから三角フィルタを並べます。

```javascript
function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }

// メル周波数の等間隔点を計算
const melPoints = Array.from({ length: numMelBins + 2 }, (_, i) =>
  melMin + i * (melMax - melMin) / (numMelBins + 1)
);
const hzPoints = melPoints.map(melToHz);
const binPoints = hzPoints.map(h => Math.floor((frameSize + 1) * h / sr));

// 三角フィルタの構築
const filterbank = Array.from({ length: numMelBins }, (_, m) => {
  const filt = new Float32Array(frameSize / 2 + 1);
  for (let k = 0; k <= frameSize / 2; k++) {
    if (k >= binPoints[m] && k <= binPoints[m + 1])
      filt[k] = (k - binPoints[m]) / (binPoints[m + 1] - binPoints[m]);
    else if (k > binPoints[m + 1] && k <= binPoints[m + 2])
      filt[k] = (binPoints[m + 2] - k) / (binPoints[m + 2] - binPoints[m + 1]);
  }
  return filt;
});
```

### DCT-IIでMFCCを計算

メルフィルタバンクのエネルギーにDCT-IIを適用してMFCCを求めます。

```javascript
// DCT-II
for (let c = 0; c < numCoeffs; c++) {
  let sum = 0;
  for (let m = 0; m < numMelBins; m++) {
    sum += melEnergies[m] * Math.cos(Math.PI * c * (m + 0.5) / numMelBins);
  }
  coeffs.push(sum);
}
```

Pythonで書けば`librosa.feature.mfcc()`の1行で済む処理ですが、内部で何をやっているかをJavaScriptで一から実装することで理解が深まりました。

### デルタMFCCの計算

デルタMFCCは隣接フレームの差分から求めます。

```javascript
function computeDelta(matrix, numCoeffs, numFrames, N = 2) {
  let denom = 0;
  for (let n = 1; n <= N; n++) denom += 2 * n * n;

  return matrix.map((_, t) => {
    const row = new Array(numCoeffs).fill(0);
    for (let c = 0; c < numCoeffs; c++) {
      let num = 0;
      for (let n = 1; n <= N; n++) {
        const t1 = Math.min(t + n, numFrames - 1);
        const t2 = Math.max(t - n, 0);
        num += n * (matrix[t1][c] - matrix[t2][c]);
      }
      row[c] = num / denom;
    }
    return row;
  });
}
```

2次デルタMFCCはデルタMFCCにもう一度同じ関数を適用するだけで求まります。

### ヒートマップをCanvasで描画

Chart.jsではヒートマップが標準でサポートされていないため、Canvas APIで直接描画しました。値を色にマッピングする関数を自前で実装しています。

```javascript
function valToColor(v, stops) {
  const t = Math.max(0, Math.min(1, v));
  const seg = (stops.length - 1) * t;
  const i = Math.floor(seg), f = seg - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return `rgb(
    ${Math.round(a[0] + (b[0] - a[0]) * f)},
    ${Math.round(a[1] + (b[1] - a[1]) * f)},
    ${Math.round(a[2] + (b[2] - a[2]) * f)}
  )`;
}
```

MFCCは正負の値を持つので中央を黒にした発散型カラーマップ（赤→黒→青）を使い、メルスペクトログラムは低エネルギーを暗く高エネルギーを明るくする連続型カラーマップを使っています。

---

## 実装のポイントまとめ

| 処理 | 使用技術 | librosaの対応関数 |
|---|---|---|
| 音声読み込み | Web Audio API | `librosa.load()` |
| FFT | 自前実装（Radix-2） | `np.fft.fft()` |
| メルフィルタバンク | 自前実装 | `librosa.filters.mel()` |
| MFCC | DCT-II自前実装 | `librosa.feature.mfcc()` |
| デルタMFCC | 有限差分自前実装 | `librosa.feature.delta()` |
| ヒートマップ | Canvas API | `librosa.display.specshow()` |

---

## 使い方

**[SpectraLens](https://KTA805.github.io/spectralens/)** にアクセスして、音声ファイルをドロップするだけで使えます。

WAV / MP3 / OGGに対応しています。デモ信号（440Hz + 880Hz + 1760Hz + ノイズの合成信号）も用意しているので、ファイルがなくてもすぐ試せます。

ソースコードはGitHubで公開しています。
🔗 [github.com/KTA805/spectralens](https://github.com/KTA805/spectralens)

---

## おわりに

Pythonのlibrosaで当たり前のように使っていた関数を、JavaScriptでゼロから実装してみると、FFTやMFCCの内部処理への理解が格段に深まりました。

今後の予定として、SNR推定・ノイズフロア自動検出・PDF出力あたりを追加していきたいと思っています。また音声AI向けのデータセット品質チェッカーとしての用途も検討しています。

音声処理・DSP・Pythonについてはこちらでも書いています。
👉 [Zennのプロフィール](https://zenn.dev/kta805)

音声処理の基礎からMFCCまで体系的に学びたい方はこちらもどうぞ。
👉 [Pythonではじめる音声処理 実践入門（Zenn本）](https://zenn.dev/kta805/books/BOOK_SLUG)
