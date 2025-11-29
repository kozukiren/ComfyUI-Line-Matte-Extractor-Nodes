# ComfyUI Line / Matte Extractor Nodes

スキャンした線画や、白い（または黄ばんだ）紙の上に描かれたキャラクターから、
**線画抽出** と **背景透過（マット生成）** を行うための ComfyUI カスタムノードセットです。

このリポジトリには以下が含まれます：

* `line_art_extractor.py`

  * 単体画像の線画抽出ノード `LineArtExtractor`
  * 連番画像フォルダ用の線画抽出ノード `DirectoryLineArtExtractor`
  * カラーキャラ＋白背景用のマット抽出ノード `DirectoryCharacterMatteExtractor`
* `透過.json`

  * 上記ノードを使ったサンプルワークフロー（単体画像／連番線画／カラーキャラ透過）

---

## 対応環境

* ComfyUI 0.3.7 以降（デスクトップ版で動作確認）
* Python 3.10+（ComfyUI 同梱の環境でOK）
* 追加ライブラリ

  * `Pillow`
  * `numpy`
  * `opencv-python`（動画読み込みノードを使う場合）

※ 通常は `ComfyUI-Manager` で依存関係を自動インストールするか、
仮想環境上で `pip install pillow numpy opencv-python` で導入してください。

---

## インストール

1. ComfyUI の `custom_nodes` フォルダを開く

   * 例：`/Users/あなたのユーザー名/AI/custom_nodes/`

2. このリポジトリをクローンまたはコピー

   ```bash
   cd custom_nodes
   git clone https://github.com/kozukiren/ComfyUI-Line-Matte-Extractor-Nodes.git
   ```

   もしくはフォルダごと `custom_nodes` 配下へドラッグ＆ドロップ。

3. ComfyUI を再起動

4. ノード検索 (`TAB` / `右クリック → ノードを追加`) で

   * `Line Art Extractor (PNG)`
   * `Directory Line Art Extractor`
   * `Directory Character Matte Extractor`
     が追加されていれば導入完了です。

---

## 収録ノード

### 1. LineArtExtractor

**用途**
1枚の入力画像から線画を抽出し、**透過PNG 用の RGBA IMAGE** を返すノード。

**INPUT**

* `image` : IMAGE

  * ComfyUI 標準 `LoadImage` などから接続
* `threshold` : FLOAT (0–1, default 0.5)

  * どの明るさまで線として残すかのしきい値
* `invert_input` : BOOLEAN

  * 白地に黒線なら True
  * 黒地に白線など、線と背景が逆のとき False
* `median_filter` : INT (1,3,5,…) (default 3)

  * 小さなゴミをならすメディアンフィルタのサイズ

**OUTPUT**

* `line_art` : IMAGE

  * RGBA (0–1) のバッチ。線は黒、背景はアルファ0。

---

### 2. DirectoryLineArtExtractor

**用途**
PNG / JPG 連番が入ったフォルダから **全フレームの線画をまとめて抽出** し、IMAGE バッチとして返すノード。

**INPUT**

* `directory_path` : STRING

  * `0001.png, 0002.png, …` のような連番が入ったフォルダパス
* `threshold` : FLOAT
* `invert_input` : BOOLEAN
* `median_filter` : INT

（パラメータの意味は `LineArtExtractor` と同じ）

**OUTPUT**

* `line_art_frames` : IMAGE

  * 連番全体が1つの IMAGE バッチにまとまっています。
  * `SaveAnimatedPNG` や `SaveAnimatedWEBP` にそのまま接続できます。

---

### 3. DirectoryCharacterMatteExtractor

**用途**
白〜黄ばんだ紙の上に描かれた**カラーキャラクターの連番画像**から、
キャラだけ切り抜いて **背景透過PNG** にするノード。
目や服の白など「キャラ内部の白」は残しつつ、
キャラの外側に広がる紙だけを透明化します。

**INPUT**

* `directory_path` : STRING

  * PNG / JPG 連番フォルダ
* `bg_tolerance` : FLOAT (default 0.10)

  * 推定した紙色とどれくらい近い色まで紙とみなすか
* `value_min` : FLOAT (default 0.70)

  * これより暗いピクセルは紙候補にしない（影や塗りの保護）
* `saturation_max` : FLOAT (default 0.40)

  * これより彩度が高い色は紙ではなくキャラ側と判断
* `close_gaps` : INT (default 2)

  * 線のスキマを何pxぶん埋めるか（背景リーク防止）
* `edge_feather` : INT (default 2)

  * アルファのエッジをどの程度ぼかすか
* `matte_shift` : INT (default -1)

  * 抜きマットを内側/外側に何pxシフトさせるか
  * プラス = 内側に食い込む（白フチ削り）
  * マイナス = 外側に広げる（欠け防止）
* `min_foreground_area` : INT (default 35)

  * これより小さい前景の島はゴミとみなして削除
* `min_hole_area` : INT (default 10000 程度推奨)

  * キャラ内部の「紙色の穴」の最小サイズ
  * これより大きい穴（腕と体の隙間など）は背景として抜き、
    小さい穴（目のハイライト等）は残す

**OUTPUT**

* `images` : IMAGE

  * RGBA バッチ。キャラは不透明、背景紙はアルファ0。

---

## サンプルワークフロー（透過.json）

`透過.json` は、上記ノードの使い方をまとめたサンプルワークフローです。

### 内容

* グループ1：**画像一枚ずつ透過**

  * `LoadImage → LineArtExtractor → PreviewImage / SaveImage`
* グループ2：**連番線画の一括透過＆アニメ書き出し**

  * `DirectoryLineArtExtractor → SaveAnimatedPNG / SaveAnimatedWEBP`
* グループ3：**色付きキャラ＋白背景の透過**

  * `DirectoryCharacterMatteExtractor → SaveAnimatedWEBP / SaveAnimatedPNG`

各グループには日本語の `Note` ノードで使用手順が書かれているので、
ワークフローをそのまま読み込んでパラメータをいじりながら確認できます。

---

## 使い始めのおすすめ設定

### 線画抽出（LineArt系）

* `threshold` : 0.5 からスタート

  * 線が薄く消える → 0.35〜0.45 に下げる
  * ゴミが多い／ベタが残りすぎ → 0.6 前後まで上げる
* `median_filter` : 3

  * スキャン線画やアナログ原稿におすすめ

### キャラ透過（DirectoryCharacterMatteExtractor）

1. まず `min_hole_area` を大きめ（例：10000）にして
   「外側の紙だけ」きれいに抜ける設定を探す
2. 腕と体の隙間なども抜きたくなったら、
   `min_hole_area` を少しずつ下げて、
   抜きたい穴だけが引っ掛かるラインを探す
3. 紙が黄ばんでいる場合は `saturation_max` を 0.3〜0.4 に

---

## ライセンス

* コード・ワークフローともにお好きに改造・利用して構いません。
  作品内での使用時に作者表記は不要ですが、
  もしクレジット頂ける場合は `KALIN / comfyui-line-matte-extractor` としていただけると嬉しいです。

---

## 作者

* KALIN

フィードバックや改善案があれば X　https://x.com/kozukiren　でぜひ教えてください。
