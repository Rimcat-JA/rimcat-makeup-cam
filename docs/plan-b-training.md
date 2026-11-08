# 案B の学習ガイド

学習モデルの仕上げ段階（`neural` エフェクト）で使うモデルを作る手順です。
守る規則は [`docs/dev/contract-plan-b.md`](dev/contract-plan-b.md) にあり、このガイドとずれたときは契約が優先です。

**現時点で本物の学習データはありません。** この文書の中の学習結果はすべて、古典パイプラインの出力を目標にした代用品のペアで得たものです。
配線が通ることを示すだけで、学習した見た目がよいかどうかについては何も示しません。

---

## 1. 全体の流れ

| 段階 | コマンド | 環境 | 出力 |
|---|---|---|---|
| ペアを作る | `python -m training.pairs ...` | `.venv-neural` | データのフォルダ |
| 教師を学習 | `python -m training.train --config ...` | `.venv-train`（借りた GPU も可） | チェックポイント |
| 生徒を蒸留 | `python -m training.train --config ... --teacher ...` | 同上 | チェックポイント |
| 書き出しと確認 | `python -m training.export ...` | `.venv-train` | `model.onnx`、`model.json`、`export_check.json` |
| 未学習の人で比較 | `python -m training.heldout ...` | どちらでも | 数値と見比べ用の画像 |
| 遅延を測る | `python -m training.latency ...` | `.venv-neural` | 数値 |
| 評価 | `python -m makeup_cam.evaluate ...` | `.venv-neural` | `evaluation.json`。合格しないモデルは通話に使えません |

コマンドはすべてリポジトリのいちばん上のフォルダで実行します。アプリは `training/` を読み込みません。何も入れなくてもアプリの動きは変わりません。

教師は省けます。`--teacher` なしで生徒だけを学習できます（§5 の表の「教師なし」）。最初の確認は §4.1 の手順がいちばん早いです。

この文書のコマンドの中の大文字は、自分で決めるフォルダです。

| 名前 | 意味 | 例 |
|---|---|---|
| `DATA` | ペアのフォルダ（`training.pairs` の `--out`） | `.test-media/planb/pairs/natural-polished` |
| `RUN` | 学習の記録とチェックポイントのフォルダ | `.test-media/planb/runs/smoke` |
| `OUT` | 書き出したモデルの置き場所 | `.test-media/planb/models` |
| `NAME` | 未学習の人を選ぶパターン（シーケンス名に対して） | `cremad-1060-*` |

### 環境を作る

```bash
py -3.12 -m venv .venv-train
.venv-train/Scripts/python -m pip install -r training/requirements.txt
py -3.12 -m venv .venv-neural
.venv-neural/Scripts/python -m pip install -r requirements-neural.txt
```

文字コードの問題を避けるため、先に `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8` を実行します。

### 置く場所の決まり

- データ、チェックポイント、モデルは、Git で管理する場所に置きません。`training.pairs` と `training.export` は、そうした場所を指定すると止まります。
- 本物のデータは暗号化したドライブに置きます（撮影ガイド 7.2）。
- 設定ファイルには絶対パス、ユーザーのフォルダ、メールアドレスを書けません（書くと止まります）。フォルダはコマンドの引数で渡します。

---

## 2. データ

### 2.1 形式

契約 §10 のとおりです。シーケンスごとに 1 フォルダで、連続したフレームが入ります。

```
<root>/<sequence>/manifest.json
<root>/<sequence>/input/000000.png     入力側。input_preset の古典パイプラインを通したもの
<root>/<sequence>/target/000000.png    メイク側。input と 1 画素単位で位置がそろったもの
<root>/<sequence>/landmarks.npy        (N, 478, 2) float32、入力側のランドマーク
<root>/<sequence>/masks/000000.png     任意。髪と顔の肌のマスクの和（uint8）
```

- シーケンス名の最後の `-` より前を「同じ人」とみなします（`cremad-1002-IEO_HAP_HI` なら `cremad-1002`）。未学習の人に指定すると、その人のシーケンスはすべて学習から外れます。名前に個人名を入れないでください。
- `manifest.json` に `"consecutive": false` があるシーケンスは、時間方向の損失を使いません。

### 2.2 作り方

**代用品（配線の確認専用）**：CREMA-D の女性を古典パイプラインに通します。入力は既定で `off`、目標は `natural` です。`demographics.csv` で女性だけを選びます。

```bash
.venv-neural/Scripts/python -m training.pairs standin --out .test-media/planb/pairs/cremad-natural
# natural の上で動かすモデルを試すなら、入力側も natural にします
.venv-neural/Scripts/python -m training.pairs standin --input-preset natural --target-preset polished --out .test-media/planb/pairs/natural-polished
```

このデータで学習したモデルはつねに `stand_in: true` になり、通話には使えません。

**入力のプリセットをそろえてください。** モデルは `input_preset` の出力に足す差分を学びます。`off` から `natural` を学んだモデルを `natural` の上で動かすと、メイクが二重にかかります（眉が濃く、唇が赤くなり、チークが増えます。実際に確かめました）。一方、遅延の予算は `natural` が前提です（契約 §7.3）。評価ツールは既定で `input_preset` で測りますが、実時間の計測は `natural` でも必ず測ります。

**3D の描画など、位置のそろった素材**（仕様書 `docs/3d-model-commission-spec.md`）：入力側を古典パイプラインに通し、目標側はそのまま使います。

```bash
.venv-neural/Scripts/python -m training.pairs render --input BARE --target MADEUP --name 2026-12-01_hdri03-take01 \
    --kind 3d_render --input-preset natural --out DATA
```

**録画**（撮影ガイド 5 章）：すっぴんとメイクの録画は時間がそろっていないので、すっぴんの各フレームに、姿勢と表情がいちばん近いメイクのフレームを選びます。目尻、目頭、鼻の付け根などで相似変換をかけて重ね、全ランドマークの誤差の p95 が顔の大きさの 0.015 倍を超えるものは捨てます。色は顔の外側の画素から合わせます。

```bash
.venv-neural/Scripts/python -m training.pairs recording --capture CAPTURE --input-preset natural --out DATA
```

- 使うのは `sessions/` の中で両方の条件がそろった日だけです。テイクは撮影ガイドの規則どおり、番号がいちばん大きい `completed` のものです。`eval/` と `mock` は使いません（評価用に分けておくためです）。
- 位置合わせは相似変換だけなので、表情が少し違うペアでは輪郭がずれます。**この経路は合成したテストデータでしか動かしていません。** 実際の録画で何割のフレームが残るか、見た目に足りるかは未確認です。最初に見比べ用の画像で確かめてください。
- 前提：`input_preset` は顔の形を変えないものに限ります（`strong` は止まります）。

---

## 3. 何を学習するか

モデルは「古典パイプラインの結果に足す差分」を出します。入力は BGR、[0, 1] の `extract_crop` のクロップです。強さのスライダーはモデルに渡しません。差分に掛けるだけです。

### 3.1 損失

重みは設定の `loss` で変えられます。下の表は初期値です。すべて `.venv-train` の CPU と `torch-directml` で動きます（FFT と複素数は使いません）。

| 項目 | 重み | 内容 |
|---|---|---|
| `lab` | 1.0 | Lab での L1。入力 + 差分 と 入力 + 目標の差分 を比べます。§5 の `weight` と端の窓でマスクします |
| `lab_partial` | 0.5 | 同じものを、0.25〜1 のランダムな強さで。スライダーの途中でも目標との比が崩れないようにします |
| `detail` | 0.2 | 輝度の勾配の L1（2 段階の解像度）。アイラインのような線が平均でぼやけるのを防ぎます。学習済みの重みは不要です |
| `temporal` | 0.5 | 隣のフレームとの差分の変化が、目標の変化より大きい分への罰。クロップで合わせたあと、残りの動きをオプティカルフローで合わせます。フローが外れた画素は軽く扱います |
| `lag` | 0.1 | 実行時は通常 1 フレーム前の差分を貼るので、差分が次のフレームの目標にも合うようにします |
| `region` | 5.0 | `weight` が 0 の場所（口の内側、目の開口部、顔と髪の外）で差分を 0 にします。ランタイムのマスクに頼らないためです |
| `magnitude` | 0.05 | 同一性の保護：差分の大きさを抑えます。`max_abs_delta` を超えた分は強く罰します |
| `low_frequency` | 0.2 | 同一性の保護：目標にない細かい差分を罰します |
| `perceptual` | 0（切） | VGG16 の特徴量。下の注意を参照 |
| `identity` | 0（切） | 顔認識モデルの埋め込みの一致。下の注意を参照 |
| `distill` | 1.0 | 生徒の差分と教師の差分の L1（`--teacher` のときだけ） |

**学習済みの重みについて**：`perceptual` と `identity` は、手元のファイルを `--perceptual-weights`、`--identity-weights` で渡したときだけ働きます。ダウンロードはしません。重みの付いた設定でファイルを渡さないと止まります。今回はどちらも使っていません。torchvision の VGG16 の重みには明確な使用条件の記載が見当たらず（ImageNet 由来）、確認できていません。使う前に、出所のライセンスを自分で確認してください。

### 3.2 データの水増し

| 種類 | 内容 | どこにかけるか |
|---|---|---|
| 部屋の光 | 露出 ±0.5 段、ホワイトバランス ±8 %、ガンマ ±15 %、顔を横切る明るさの傾き | 入力と目標の両方に同じ値を。隣のフレームにも同じ値を |
| カメラ | センサーのノイズ、軽いぼけ、JPEG | 入力だけ。目標の差分はきれいなまま |
| クロップ | ランドマークのずれ（並進、回転、拡大）を両フレームに同じだけ | 両方 |

カメラの劣化を入力だけにかけるのは、モデルがノイズを消したり写したりしないようにするためです。左右反転は使いません（顔と髪の分け目は左右対称ではありません）。

### 3.3 モデル

`training/models.py` の U-Net で、大きさは設定の `model` で決めます。最後の層はちょうど 0 で始まるので、学習前のモデルは何も変えません。活性化関数は既定で LeakyReLU です（代用品のペアで、正規化なしの ReLU の生徒は出力が定数になって止まりました。§5）。

- 生徒：契約 §8 の演算だけを使います（`Conv`、`LeakyRelu`、`Resize`（linear）、`Concat`、`Tanh`、必要なら `InstanceNormalization`）。目標は `student-512.json` の 8.6 MB、512（45 GMAC）です。`float16` を指定すると、グラフの内側だけ float16 にします。**大きさの上限はここです。** InstanceNorm の生徒を実際に測ると、p32 @512 float16 が往復 p95 で 6〜8 ms、p40 @512 からは 16 ms 以上で予算の 12 ms を超えます（契約 §1.2）。
- 教師：残差ブロックと self-attention を使えます。書き出しません。

---

## 4. 学習と書き出し

```bash
# 教師
.venv-train/Scripts/python -m training.train --config training/configs/teacher-rented.json --data DATA --run-dir RUN/teacher --device cuda
# 生徒（蒸留）
.venv-train/Scripts/python -m training.train --config training/configs/student-512.json --data DATA --run-dir RUN/student \
    --teacher RUN/teacher/ckpt/last.pt --device cuda
# 書き出し
.venv-train/Scripts/python -m training.export --checkpoint RUN/student/ckpt/last.pt --data DATA --out OUT/finish-student-0.1.0
```

- **設定ファイル**：1 回の学習に 1 つです。`RUN/run_config.json` に写しが残ります。`training/configs/` の `teacher-rented.json` と `student-512.json` は出発点で、本物のデータではまだ動かしていません。`smoke-*.json` は配線の確認用です。
- **再開**：同じコマンドに `--resume` を付けます。途中で止まっても、止まらなかった場合と同じ重みになります（テストで確認しています）。乱数はすべて（seed、ステップ）から作るので、覚えておく状態がありません。変えてよいのは `steps` と記録の間隔だけで、ほかを変えると止まります。
- **未学習の人**：`data.heldout_sequences` に指定します。`validate_every` ごとに、モデルの Lab 誤差と何も変えない場合の Lab 誤差を比べて `metrics.jsonl` に記録します。比が 1 を下回らないモデルは先に進めません。
- **書き出しの確認**（契約 §8）：ONNX Runtime（CPU）で読み込み、形、1 ファイルであること、演算の許可リスト、opset 17〜21、実際のクロップでの PyTorch との差を確かめます。差の許容は float32 が 1e-3、float16 が 5e-3 です。1 つでも外れるとモデルを消して止まります。
- **`stand_in`**：学習データの種類から自動で決まります。`3d_render` と `recording` だけで学習したときだけ `false` になります。代用品が 1 本でも混ざると `true` です。
- **既にあるフォルダ**：`training.export` は同じフォルダのモデルを上書きしません。`--overwrite` を付けると置き換えます。その場合、前の評価の結果は使えなくなります。
- **`--device` の既定は `cpu` です。** `cuda` は借りた GPU、`dml` は `torch-directml` の環境で使います。
- **`stand_in` の根拠**：`training.pairs render --kind` に書いた種類をそのまま信じます。古典パイプラインの出力を `--kind recording` と書くと `stand_in: false` になってしまうので、本当の素材にだけ使ってください。ツールには見分けられません。
- **通話で使う前に**：評価に合格する必要があります。合格には次の全部が要ります。`--image` と `--video` の `--neural-model` は評価なしで動きますが、仮想カメラには出ません。

```bash
.venv-neural/Scripts/python -m makeup_cam.evaluate --model OUT/finish-student-0.1.0 --measure-fps --soak-minutes 30 \
    --identity-model 顔認識モデル.onnx --identity-licence "ライセンス"
```

```bash
.venv-train/Scripts/python -m training.heldout --model OUT/finish-student-0.1.0 --data DATA --sequences "NAME-*" --out OUT/report.json --sheet OUT/sheet.png
.venv-neural/Scripts/python -m training.latency --model OUT/finish-student-0.1.0
```

### 4.1 手元で最初から最後まで通す（代用品）

借りた GPU なしで、ペア作成から評価まで 1 時間半ほどで通ります。見た目については何も分かりません。配線の確認です。

```bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
DATA=.test-media/planb/pairs/natural-polished
# 1. ペア（.venv-neural、10 分ほど。入力は natural、目標は polished）
.venv-neural/Scripts/python -m training.pairs standin --input-preset natural --target-preset polished --out $DATA
# 2. 生徒を学習（.venv-train、CPU で 2,500 ステップ、20〜40 分。教師なし）
.venv-train/Scripts/python -m training.train --config training/configs/smoke-student.json --data $DATA --run-dir .test-media/planb/runs/smoke --device cpu
# 3. 書き出しと確認（.venv-train、数秒。float16 のモデルは確認に数分かかることがあります）
.venv-train/Scripts/python -m training.export --checkpoint .test-media/planb/runs/smoke/ckpt/last.pt --data $DATA --out .test-media/planb/models/smoke-student-0.0.1
# 4. 未学習の人と遅延
.venv-train/Scripts/python -m training.heldout --model .test-media/planb/models/smoke-student-0.0.1 --data $DATA --sequences "cremad-1060-*" --out .test-media/planb/heldout.json --sheet .test-media/planb/heldout.png
.venv-neural/Scripts/python -m training.latency --model .test-media/planb/models/smoke-student-0.0.1
# 5. 動画 1 本に通す（評価なしで動きます。仮想カメラには出ません）
.venv-neural/Scripts/python -m makeup_cam --video 入力.flv --out 出力.mp4 --preset natural --neural-model .test-media/planb/models/smoke-student-0.0.1
# 6. 評価（9 分ほど。30 分の実行を足すと 40 分ほど）
.venv-neural/Scripts/python -m makeup_cam.evaluate --model .test-media/planb/models/smoke-student-0.0.1 --measure-fps --out .test-media/planb/evaluate/smoke
```

`smoke-student.json` のデータの指定（未学習の人）は、シーケンス名が `cremad-` で始まることを前提にしています。

時間は他の作業の量で大きく変わります。例えば生徒 p16 の 2,500 ステップは、他の作業が少ないときに約 20 分、別の学習と同時で 42 分でした。ペアの作成は 505〜568 秒、float16 の書き出しの確認は p32 で約 2 分、p96 で約 26 分でした（他の作業と同時）。float32 の書き出しは数秒です。

### 4.2 モデルをアプリに入れる

1. 入れる環境を選びます。`onnxruntime-directml` は `.venv-neural` にあります。`start_windows.bat` は `.venv` を使うので、学習モデルの仕上げを使うときは `.venv-neural/Scripts/python -m makeup_cam` で起動します。
2. モデルのフォルダ（`model.onnx` と `model.json`、合格していれば `evaluation.json`）をそのまま `neural_models` の下にコピーします。場所は `python -m makeup_cam.neural` が表示します（`AI_MAKEUP_CAM_NEURAL_DIR` で変えられます）。
3. `.venv-neural/Scripts/python -m makeup_cam.neural --use <名前>-<バージョン>` で選びます。一覧には、それぞれ使えるかどうかが出ます。
4. 設定の「学習モデルの仕上げ」をオンにします。

**代用品（`stand_in: true`）と評価に合格していないモデルは、選べても動きません。** 「評価に合格したモデルがありません」と表示され、映像は従来のメイクのままです。`--use` が効くのは GUI と `--headless` だけで、`--image` と `--video` は毎回 `--neural-model` で指定します。

---

## 5. 配線の確認（スモーク）の結果

**ここにある数字は配線が通ることだけを示します。** 目標が古典パイプラインの出力なので、学習できても「アプリがすでにしていることを、ぼかして真似できた」というだけです。学習した見た目のよさについては何も示しません。

### 条件

- 2026-11-06、Windows 11、Ryzen 7 9800X3D、RX 9070 XT。学習は `.venv-train` の CPU（`torch` 2.15.0、8 スレッド）です。ほかの作業が同時に動いていたので、時間は幅をもって読んでください。
- データ：CREMA-D の女性 11 人の 44 クリップを `natural` に通した 2,778 ペア。学習は 8 人（2,044 ペア）です。1060、1072、1082 の 3 人は学習に使っていません。
- 大きさはすべて S = 256。設定は `training/configs/smoke-*.json` です。

### 学習

| モデル | 重み | ステップ | 1 ステップの中央値 | 損失（最初の 50 ステップ → 最後の 50 ステップ） |
|---|---|---|---|---|
| 教師（残差、attention） | 18.0 MB | 1,500 | 1,137 ms | 0.0158 → 0.0064 |
| 生徒（教師あり） | 2.15 MB | 2,500 | 742 ms | 0.0233 → 0.0083 |
| 生徒（教師なし、比較用） | 2.15 MB | 2,500 | 491 ms | 0.0154 → 0.0067 |

### 未学習の 3 人での比較

値は、目標との Lab の平均絶対差（cv2 の尺度）です。モデルが完全に変えてよい画素だけで測っています。「比」は モデル ÷ 何も変えない場合 で、小さいほど目標に近いことを示します。

| モデル | 何も変えない | モデル | 比の平均 | 比の最悪（人） | 測り方 |
|---|---|---|---|---|---|
| 生徒（教師あり）を書き出した ONNX | 1.309 | 0.541 | **0.418** | 0.555（1072） | ONNX Runtime CPU、シーケンスごとに 12 フレーム |
| 同じ ONNX | 1.304 | 0.530 | 0.412 | 0.545（1072） | DirectML、シーケンスごとに 3 フレーム |
| 教師 | 1.310 | 0.651 | 0.507 | 0.671（1072） | 学習中の検証、6 フレーム |
| 生徒（教師なし） | 1.310 | 0.519 | 0.402 | 0.469（1072） | 学習中の検証、6 フレーム |

- 3 人とも、何も変えない場合より目標に近づきました。
- **今回は蒸留の効果がありませんでした。** 教師のほうが未学習の人で悪く、学習を続けるほど比が上がりました（0.455 → 0.507。学習した 8 人に寄っているためと考えています）。短い学習で小さい教師なので、蒸留の良し悪しはこの数字からは言えません。経路が動くことだけを確かめました。
- 強さ 0.25、0.5、1.0 で、変化の量は単調に増え、どの強さでも目標との差は何も変えない場合の半分以下でした。
- フレーム間の差分の変化：モデル 0.0011、目標 0.0015（平均絶対差）。これは本当のちらつきの評価ではありません。
- `weight` が 0 の場所の、マスクをかける前の差分：平均 0.0019、p99 0.029。ランタイムがここを 0 にするので、画面には出ません。

### 実際のランタイムを通した確認

書き出したモデルを、本物のワーカープロセス、DirectML、クロップ、貼り戻しに通しました（`off` プリセット、強さ 1、`sync=True`、未学習の 3 人の 1 クリップずつ）。

| クリップ | 比 | 古典の変化との相関 | BGR の平均変化（古典 → モデル、uint8 の段階） |
|---|---|---|---|
| 1082_IEO_HAP_HI | 0.474 | 0.889 | (3.56, 1.88, 2.98) → (3.96, 1.70, 2.16) |
| 1060_ITS_HAP_XX | 0.339 | 0.871 | (6.14, 3.75, 5.67) → (5.56, 3.67, 5.08) |
| 1072_DFA_NEU_XX | 0.587 | 0.892 | (1.75, -1.08, 0.26) → (4.26, 1.92, 2.54) |

相関が高く、位置も色の並びも合っていることから、学習側とランタイムのクロップと BGR が一致していることがわかります。1072 では全体に明るくしすぎています。評価ツール（`makeup_cam.evaluate`）も、このモデルを読み込んで動きました（40 フレームだけの確認で、判定には足りません）。

### 遅延（モデル単独、DirectML、`onnxruntime-directml` 1.24.4）

30 Hz ペース、300 回、入力はランダムな数字です。アプリは同時に動かしていません。全ノードが DirectML で動きました。

| モデル | ファイル | 作成 | 初回 | 中央値 | p95 | p99 | 最大 | プロセスの VRAM |
|---|---|---|---|---|---|---|---|---|
| 生徒 p16、instance norm、@256、float32 | 2.26 MB | 423 ms | 8.0 ms | 1.19 ms | 1.91 ms | 2.20 ms | 2.67 ms | 121 MB |
| 生徒 p32、instance norm、@512、float16 | 4.41 MB | 454 ms | 14.6 ms | 4.23 ms | 5.03 ms | 7.45 ms | 7.52 ms | 331 MB |

2 行目は `torch-directml` の一時的な環境で 400 ステップだけ学習したものです（比の平均 0.729）。遅延の確認用で、S = 512 の経路（学習、float16 の書き出し、DirectML）が通ることを示します。

### 書き出しの確認

ONNX Runtime と PyTorch の最大差：float32 @256 で 3.7e-6（許容 1e-3）、float16 @512 で 8.7e-5（許容 5e-3）。実際のクロップで測りました。演算は `Cast`（float16 のみ）、`Concat`、`Conv`、`InstanceNormalization`、`LeakyRelu`、`Resize`、`Tanh` です。

### 途中でわかったこと

- 正規化なしの ReLU の生徒は、`region` 損失に引かれて出力が定数になり、800 ステップ動かしても比が 1.000 のままでした。LeakyReLU にすると学習が進みました。
- 正規化なしの生徒は学習率 1e-3 で不安定でした（損失が一時的に 3 倍）。1,500 ステップで比べると、学習率 3e-4 で正規化なしは 0.550、学習率 1e-3 で instance norm は 0.415 でした（どちらも `torch-directml`、シーケンスごとに 2 フレームの検証）。そのため生徒の設定は instance norm にしました。

---

## 6. 必要なデータと時間

**測定**と書いた数字以外は見積もりです。実データでは 1 回も動かしていないので、桁が合っていれば十分という程度に読んでください。

### 6.1 データ量

| 項目 | 数字 | 根拠 |
|---|---|---|
| 撮影ガイドどおりの録画 | すっぴん 35 分 + メイク 35 分。すっぴん約 63,000 フレーム | 撮影ガイド 5.1 から計算 |
| 録画の容量 | 片方 1 回分で 35〜45 GB、全部で 200 GB 前後 | 撮影ガイドの測定値（ノイズを足したテスト映像）から |
| 代用品のペアの容量 | 44 シーケンス、2,778 ペア、2.77 GB（1 ペア約 1.0 MB、マスク込み） | **測定**。低解像度を拡大した素材なので、実際のカメラより小さく出ます |
| 実カメラの PNG | 1 枚 1〜2 MB、3 フレームごとに使うと 21,000 ペアで 40〜80 GB | 見積もり |
| 録画ペアで残る割合 | 不明 | 合成データで 20/36 ですが、本物の録画とは条件が違います |
| 学習に足りる枚数 | 最初の確認は数千ペア、仕上げは 2 万ペア以上 | 見積もり。測定ではありません。評価の結果で判断します |

### 6.2 1 ステップの時間

すべて**測定**です。データは代用品、ほかの作業が動いている中で、数十〜数千ステップの中央値です。バッチ 4 は 8 フレーム（1 サンプルが隣り合う 2 フレームのため）にあたります。

| モデル | S | バッチ | CPU（`.venv-train`、8 スレッド） | `torch-directml`（RX 9070 XT） |
|---|---|---|---|---|
| 生徒 p16（2.15 MB） | 256 | 4 | 491 ms | 105 ms |
| 同上、小さい教師で蒸留 | 256 | 4 | 742 ms | 測っていません |
| 生徒 p32（8.6 MB、`student-512.json` の大きさ） | 512 | 4 | 5.3 s | 630 ms |
| 教師（18 MB、`smoke-teacher.json`） | 256 | 4 | 1.14 s | 測っていません |
| 教師（288 MB、`teacher-rented.json` の大きさ） | 512 | 4 | 測っていません | **VRAM 不足で動きません** |
| 同上 | 512 | 2 | 測っていません | 1.3〜1.7 s（12 ステップだけ） |

1 サンプルを作る時間（1 スレッド）：S = 256 で約 110 ms、S = 512 で約 170 ms。多くは PNG の読み込みです。`torch-directml` は `torch` 2.4.1 に固定なので、`.venv-train` には入れられません。`grid_sample` は CPU に回ると警告が出ます。

### 6.3 全体の時間

| 作業 | 見積もり | 根拠 |
|---|---|---|
| 3D の描画 | 1 フレーム数秒〜数十秒、2 万フレームで数十時間 | 見積もり。シーンと GPU で大きく変わります |
| ペアの作成 | CREMA-D の 2,778 ペアで 505 秒（約 0.18 秒/フレーム、クリップごとのモデル読み込み込み）。2 万ペアで約 1 時間、録画 35 分分で数時間 | **測定**した速度から計算 |
| 教師（`teacher-rented.json`、100,000 ステップ、バッチ 4、A100） | 6〜20 時間 | 見積もり。下の注 |
| 生徒（`student-512.json`、100,000 ステップ、バッチ 8、A100） | 5〜10 時間 | 見積もり。下の注 |
| CPU で同じ生徒（バッチ 4） | 約 6 日 | 測定した 1 ステップから計算。現実的ではありません |
| `torch-directml` で同じ生徒（バッチ 4） | 約 18 時間 | 測定した 1 ステップから計算。長時間の安定性は未確認です |
| `torch-directml` で同じ教師（バッチ 2） | 約 2 日 | 測定した 12 ステップから計算。長時間の安定性は未確認です |

注：A100 の速さは ResNet-50 の公開値（AMP で毎秒約 1,400 枚）から畳み込みの GMAC の比で換算しただけで、測っていません。データの読み込みが律速になることもあります（S = 512 の 1 サンプルが 1 スレッドで約 170 ms なので、8 ワーカーでバッチ 8 に約 170 ms）。教師が A100 のメモリに入るかどうかも未確認です（手元の 16 GB では `torch-directml` でバッチ 4 が入りませんでした。AMP なしの値です）。最初の 1〜2 時間で実際の速さを測ってから、全体の時間を決めてください。

---

## 7. GPU を借りるときのデータの扱い

これらのデータは顔の映像です（撮影ガイド 7.2）。**借りる前に次の条件を確認し、1 つでも満たせないなら借りないでください。** このリポジトリのコードは、外部へ何も送りません。クラウドの SDK、ホスト名、鍵も入っていません。

### マシンの条件

- **データセンター型のマシンだけを使います**（vast.ai なら Secure Cloud）。個人が貸しているマシンは使いません。貸し手はマシンの中身を読めるからです。
- 運営会社と施設の認証（ISO 27001 など）を、借りる前に確認します。
- 使っている間だけ起動し、共有ストレージや公開スナップショットは使いません。

### 手順

1. **手元で暗号化します。** データとチェックポイントは、アップロードする前に手元で暗号化します。例：`tar -cf - DATA | age -p -o data.tar.age`、または 7-Zip の AES-256（ファイル名も暗号化）。パスフレーズはマシンにもリポジトリにも保存しません。
2. **SSH で送ります。** 公開鍵認証にして、パスワード認証を切ります。`scp` か `rsync -e ssh` を使います。クラウドストレージを経由しません。公開 URL も作りません。
3. **復号はマシンの中だけで、できればメモリ上に。** `/dev/shm` に入る量ならそこに展開します。入らなければインスタンスのローカルディスクに展開します。ネットワークストレージには展開しません。
4. **外への送信を切ります。** 実験管理サービス（W&B など）を使いません。学習中はできればファイアウォールで外向きの通信を止めます。`training/` はこれらを使いません。
5. **学習は `--resume` を付けて実行します。** マシンが止まっても続けられます。`--device cuda` で動きます。環境は `training/requirements.txt` の torch を、同じ版の CUDA 用に替えたものです。
6. **結果は暗号化して持ち帰ります。** `model.onnx`、チェックポイント、`metrics.jsonl` だけです。重みも学習データから作られたものとして扱います。
7. **終わったらすぐ消します。** データを消し、インスタンスを**削除**します（停止では不十分です）。ボリュームやスナップショットが残っていないことを管理画面で確かめ、日付とインスタンスの種類を手元のメモに残します。マシンの中で `shred` をかけても、仮想ディスクでは確実に消せないので、頼りません。

### 借りる前に手元で確かめること

- 本物のデータで `smoke-student.json` と同じ規模の学習を手元で流し、未学習のセッションで比が 1 を下回り、見比べ用の画像に破綻がないこと。
- `python -m pytest tests/test_training_core.py tests/test_training_torch.py` が借りるマシンでも通ること（データは要りません）。これは顔のデータを送る前に行います。

---

## 8. いつ借りる価値があるか

| 状況 | 判断 |
|---|---|
| 本物のペアがまだない | **借りません。** 学習するものがありません |
| コードの確認、代用品での確認 | 借りません。CPU で足ります（§5） |
| 生徒だけを 256 で試す、数千ステップ | 借りません。CPU か `torch-directml` で足ります |
| 本物のペアで教師を試すが、ペアの質が未確認 | 借りません。先に手元で小さい学習をして、見比べ用の画像で確かめます。データの問題に時間料金を払うことになります |
| 手元の小さい学習で比が 1 を下回り、破綻もない。次は 512 の教師 | **借ります。** 手元では CPU だと現実的な時間で終わらず、`torch-directml` だとバッチ 2 で約 2 日かかります。まず 1〜2 時間で速さを測ります |
| 同じ設定を何度もやり直す段階 | 1 回の時間と回数を掛けて、手元の `torch-directml` と比べます。ただし `torch-directml` は torch 2.4.1 に固定で、長時間の安定性は未確認です |

判断の目安：1 回の学習が手元で 1 晩（約 10 時間）に収まらず、データの質を手元で確認済みのときに借ります。料金は変わるので、借りる時点の時間単価に §6.3 の時間を掛けて確かめてください。

---

## 9. 確認していないこと

- 本物のペアでの学習。見た目、ちらつき、同一性は何も確認していません。
- `--device cuda`（CUDA の実機がないため）。コードの経路はありますが、動かしていません。
- 録画ペアの経路の実データでの動作。
- `perceptual` と `identity` の実際の重みでの学習（テストでは偽物と、ランダムな重みの VGG16 で動作だけを確かめました）。
- 長時間の学習の安定性。いちばん長い実行は CPU で 2,500 ステップ（約 31 分）、`torch-directml` で 1,500 ステップです。
- 横を向いた顔。CREMA-D はほぼ正面です。
