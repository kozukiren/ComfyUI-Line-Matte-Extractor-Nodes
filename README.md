# ComfyUI Line / Matte Extractor Nodes

スキャンした線画や、白い（または黄ばんだ）紙の上に描かれたキャラクター画像から、

* **線画抽出（Line Art Extraction）**
* **背景透過（マット生成 / Matte Extraction）**

を行うための **ComfyUI カスタムノードセット**です。

---

## このリポジトリに含まれるもの

### 1) 線画抽出（line_art_extractor.py）

* **LineArtExtractor**：単体画像の線画抽出
* **VideoLineArtExtractor**：動画から線画の連番（PNG）生成
* **DirectoryLineArtExtractor**：連番画像フォルダの線画抽出

### 2) キャラ背景透過（character_matte_extractor.py）

* **CharacterMatteExtractor**：単体画像の背景透過
* **DirectoryCharacterMatteExtractor**：連番画像フォルダの背景透過

### 3) ベタ塗り化（オプション）（flat_color_posterizer.py）

* **FlatColorPosterizer**：陰影・ハイライトをならしつつ、指定色数でベタ塗り化（ポスタライズ）

### 4) UI拡張（js/character_matte_extractor.js）

* **スポイト（EyeDropper）**で `background_color` を拾えるUIを追加
  ※対応ブラウザのみ（CharacterMatteExtractor / DirectoryCharacterMatteExtractor 用）

### 5) サンプルワークフロー（透過.json）

* 単体線画／連番線画／連番キャラ透過 のサンプル

---

## 対応環境

* **ComfyUI**（デスクトップ版で動作確認）
* **Python 3.10+**（ComfyUI 同梱のPythonでOK）

### 追加ライブラリ（requirements.txt）

* **opencv-python**：動画読み込み、マット抽出で使用
* **scikit-learn**：Flat Color Posterizer の色数削減で使用
  ※ **Pillow / numpy** は多くのComfyUI環境に同梱されていますが、環境によっては別途必要です。

---

## インストール

1. ComfyUI の `custom_nodes` フォルダを開く
   例：`/Users/<ユーザー名>/AI/custom_nodes/`

2. このリポジトリをクローン

```bash
cd /Users/<ユーザー名>/AI/custom_nodes/
git clone https://github.com/kozukiren/ComfyUI-Line-Matte-Extractor-Nodes.git
```

（またはフォルダごと `custom_nodes` 配下へ配置）

3. **ComfyUI を再起動**

4. ノード検索（TAB / 右クリック → ノード追加）で以下が出ていれば導入完了

* Line Art Extractor (PNG)
* Video Line Art Extractor (PNG sequence)
* Directory Line Art Extractor (PNG sequence)
* Character Matte Extractor
* Directory Character Matte Extractor
* Flat Color Posterizer

---

# 収録ノード

## 1) Line Art Extractor (PNG)（LineArtExtractor）

**用途**：1枚の入力画像から線画を抽出し、透過PNG向けの **RGBA IMAGE** を返します。

### INPUT

* `image : IMAGE`
* `threshold : FLOAT (0–1, default 0.5)`
  どの明るさまで「線」として残すかのしきい値
* `invert_input : BOOLEAN (default False)`

  * 白背景に黒線（一般的なスキャン線画）→ **True 推奨**
  * 黒背景に白線など反転不要 → **False**
* `median_filter : INT (odd: 1,3,5,…) (default 3)`
  小さなゴミをならすメディアンフィルタ

### OUTPUT

* `line_art : IMAGE`
  RGBAバッチ。線は黒、背景はアルファ0。

---

## 2) Directory Line Art Extractor (PNG sequence)（DirectoryLineArtExtractor）

**用途**：連番画像フォルダから、全フレームの線画を一括抽出して **IMAGEバッチ**で返します。

### INPUT

* `directory_path : STRING`
  `0001.png, 0002.png, ...` のような連番が入ったフォルダ
  対応拡張子：png / jpg / jpeg / bmp / tif / tiff
* `threshold / invert_input / median_filter`
  意味は LineArtExtractor と同じ

### OUTPUT

* `line_art_frames : IMAGE`
  連番全体が1つのIMAGEバッチにまとまります。

---

## 3) Video Line Art Extractor (PNG sequence)（VideoLineArtExtractor）

**用途**：動画からフレームを抜き、線画透過の **RGBA連番**を返します。

### INPUT

* `video_path : STRING（フルパス）`
* `threshold / invert_input / median_filter`
* `sample_every : INT (default 1)`
  何フレームごとに処理するか（1=全フレーム）

### OUTPUT

* `line_art_frames : IMAGE`

---

## 4) Directory Character Matte Extractor（DirectoryCharacterMatteExtractor）

**用途**：白〜黄ばんだ紙の上に描かれた **カラーキャラの連番**から、キャラだけを切り抜いて **背景透過RGBA**にします。

### 仕組み（概要）

* `background_color` に近いピクセルを「背景候補」として抽出
* 画像の**外周とつながっている部分だけ**を背景とみなし透明化
  → 目や服の白など「キャラ内部の白」は残りやすい設計

### INPUT

* `directory_path : STRING`
  PNG/JPG連番フォルダ（.png / .jpg / .jpeg）
* `background_color : STRING (default #FFFFFF)`
  抜きたい紙の代表色（HEX）
  ※ `js/character_matte_extractor.js` が有効ならスポイト指定可能（対応ブラウザのみ）
* `threshold : FLOAT (default 0.10)`
  `background_color` との色距離がこの値以下を「背景候補」とみなす
* `close_gaps : INT (default 1)`
  輪郭の小さな隙間を埋めて背景の侵入（リーク）を抑える
* `edge_feather : INT (default 2)`
  アルファ境界をぼかす
* `matte_shift : INT (default 0)`
  マットの膨張/収縮

  * プラス：外側に広げる（欠け防止）
  * マイナス：内側に縮める（白フチ削り）
* `min_foreground_area : INT (default 16)`
  小さすぎる前景の島をゴミとして除去
* `min_hole_area : INT (default 128)`
  キャラ内部にできた「背景色の穴」のうち、**面積が大きい穴だけ**を抜くためのしきい値
  → 小さい穴（ハイライト等）は残しやすい

### OUTPUT

* `images : IMAGE`
  RGBAバッチ。キャラは不透明、背景はアルファ0。

---

## 5) Character Matte Extractor（CharacterMatteExtractor）

**用途**：DirectoryCharacterMatteExtractor の **単体画像（IMAGE入力）版**です。
パラメータは同一で、IMAGEバッチを処理して RGBA を返します。

---

## 6) Flat Color Posterizer（FlatColorPosterizer）※オプション

**用途**：陰影・ハイライトをならして、指定色数で **ベタ塗り化（ポスタライズ）**します。

### 注意

* `opencv-python` と `scikit-learn` が必要です。

### INPUT（抜粋）

* `images : IMAGE`
* `num_colors : INT (2–16, default 4)`
* `smoothing_radius : INT (default 5)`
* `edge_preserve : FLOAT (0–1, default 0.1)`
* `color1_hex` 〜 `color8_hex`（任意）
  パレット色をHEXで指定（指定分を優先して使用）

### OUTPUT

* `images : IMAGE`

---

## サンプルワークフロー（透過.json）

`透過.json` は、線画抽出（単体／連番）とキャラ透過（連番）の使い方をまとめたサンプルです。
※ワークフロー内に **Fast Groups Muter (rgthree)** が含まれるため、未導入環境ではそのノードのみ置き換え/削除してください。

---

# 使い始めのおすすめ設定

## 線画抽出（LineArt系）

* `threshold`：まず **0.5** から

  * 線が薄く消える → **0.35〜0.45** に下げる
  * ゴミが多い／ベタが残る → **0.6前後**まで上げる
* `invert_input`：

  * 白背景に黒線 → **True推奨**
* `median_filter`：**3**

  * スキャン線画・アナログ原稿におすすめ

## キャラ透過（CharacterMatte系）

1. `background_color` を紙色に合わせる（白紙以外は最重要）
2. `threshold` を調整

   * 紙が残る → 少し上げる
   * キャラの薄い色まで抜ける → 少し下げる
3. 状況別

   * 紙が輪郭内に入り込む → `close_gaps` を上げる
   * 白フチが気になる → `matte_shift` をマイナスへ（少しずつ）
   * 欠けが気になる → `matte_shift` をプラスへ（少しずつ）
   * 大きい穴も抜きたい → `min_hole_area` を下げる
   * 目ハイライト等を残したい → `min_hole_area` を上げる

---

## ライセンス

コード・ワークフローともに自由に改造・利用してOKです。
作品内での作者表記は不要ですが、もしクレジットを頂ける場合は
**KALIN / comfyui-line-matte-extractor** としていただけると嬉しいです。
