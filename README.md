# ComfyUI Line / Matte Extractor Nodes

スキャンした線画や、白い（または黄ばんだ）紙の上に描かれたキャラクターから、
**線画抽出** と **背景透過（マット生成）** を行うための ComfyUI カスタムノードセットです。 

このリポジトリには以下が含まれます：

* `line_art_extractor.py` :contentReference[oaicite:2]{index=2}
  * 単体画像の線画抽出ノード：`LineArtExtractor`
  * 動画ファイルから線画連番を生成：`VideoLineArtExtractor`
  * 連番画像フォルダ用の線画抽出：`DirectoryLineArtExtractor`

* `character_matte_extractor.py` :contentReference[oaicite:3]{index=3}
  * 単体画像から背景透過：`CharacterMatteExtractor`
  * 連番画像フォルダから背景透過：`DirectoryCharacterMatteExtractor`

* `flat_color_posterizer.py`（オプション機能） :contentReference[oaicite:4]{index=4}
  * 陰影・ハイライトをならしつつ指定色数でベタ塗り化：`FlatColorPosterizer`

* `js/character_matte_extractor.js` :contentReference[oaicite:5]{index=5}
  * `CharacterMatteExtractor` / `DirectoryCharacterMatteExtractor` の
    `background_color` を **スポイト（EyeDropper）で拾うUI** を追加します（対応ブラウザのみ）。

* `透過.json` :contentReference[oaicite:6]{index=6}
  * 収録ノードを使ったサンプルワークフロー（単体線画／連番線画／連番キャラ透過）

---

## 対応環境

* ComfyUI（デスクトップ版で動作確認）
* Python 3.10+（ComfyUI 同梱の環境でOK）
* 追加ライブラリ（`requirements.txt`） :contentReference[oaicite:7]{index=7}
  * `opencv-python`（動画読み込み・マット抽出で使用）
  * `scikit-learn`（Flat Color Posterizer の色数削減で使用）

※ `Pillow` / `numpy` は多くの ComfyUI 環境で同梱されていますが、環境によっては別途必要です。 

---

## インストール

1. ComfyUI の `custom_nodes` フォルダを開く  
   例：`/Users/あなたのユーザー名/AI/custom_nodes/`

2. このリポジトリをクローン

   ```bash
   cd custom_nodes
   git clone https://github.com/kozukiren/ComfyUI-Line-Matte-Extractor-Nodes.git
もしくはフォルダごと custom_nodes 配下へ配置。

ComfyUI を再起動

ノード検索（TAB / 右クリック → ノードを追加）で以下が出ていれば導入完了：

Line Art Extractor (PNG)

Video Line Art Extractor (PNG sequence)

Directory Line Art Extractor (PNG sequence)

Character Matte Extractor

Directory Character Matte Extractor

Flat Color Posterizer

収録ノード
1. Line Art Extractor (PNG)（LineArtExtractor）
用途
1枚の入力画像から線画を抽出し、透過PNG向け RGBA IMAGE を返します。 
line_art_extractor


INPUT

image : IMAGE

threshold : FLOAT (0–1, default 0.5)

どの明るさまで「線」として残すかのしきい値

invert_input : BOOLEAN (default False)

白背景に黒線（一般的なスキャン線画）の場合は True 推奨

黒背景に白線など、反転不要な素材は False

median_filter : INT (odd: 1,3,5,…) (default 3)

小さなゴミをならすメディアンフィルタ

OUTPUT

line_art : IMAGE

RGBA のバッチ。線は黒、背景はアルファ0。 
line_art_extractor


2. Directory Line Art Extractor (PNG sequence)（DirectoryLineArtExtractor）
用途
連番画像が入ったフォルダから 全フレームの線画を一括抽出 して IMAGE バッチで返します。 
line_art_extractor


INPUT

directory_path : STRING

0001.png, 0002.png, … のような連番が入ったフォルダパス

対応拡張子：png/jpg/jpeg/bmp/tif/tiff 
line_art_extractor


threshold / invert_input / median_filter

意味は LineArtExtractor と同じ

OUTPUT

line_art_frames : IMAGE

連番全体が1つの IMAGE バッチにまとまっています。 
line_art_extractor


3. Video Line Art Extractor (PNG sequence)（VideoLineArtExtractor）
用途
白黒動画（または線画動画）からフレームを抜き、線画透過の RGBA 連番を返します。 
line_art_extractor


INPUT

video_path : STRING（フルパス）

threshold / invert_input / median_filter

sample_every : INT (default 1)

何フレームごとに処理するか（1=全フレーム） 
line_art_extractor


OUTPUT

line_art_frames : IMAGE

4. Directory Character Matte Extractor（DirectoryCharacterMatteExtractor）
用途
白〜黄ばんだ紙の上に描かれた カラーキャラクターの連番画像から、
キャラだけ切り抜いて 背景透過 RGBA にするノードです。 
character_matte_extractor


このノードは「背景色（紙色）に近いピクセル」を背景候補として抽出し、
画像の外周から繋がっている部分だけを背景として透明化します。
そのため、目や服の白など「キャラ内部の白」は残しやすい設計です。 
character_matte_extractor


INPUT

directory_path : STRING

PNG/JPG 連番フォルダ（.png/.jpg/.jpeg） 
character_matte_extractor


background_color : STRING (default #FFFFFF)

抜きたい紙の代表色（HEX）

js/character_matte_extractor.js が有効ならスポイトで指定できます（対応ブラウザのみ）。

threshold : FLOAT (default 0.10)

background_color との色距離がこの値以下を「背景候補」とみなします。 
character_matte_extractor


close_gaps : INT (default 1)

キャラ輪郭の小さな隙間を埋めて、背景の侵入（リーク）を抑えます。 
character_matte_extractor


edge_feather : INT (default 2)

アルファ境界をぼかして馴染ませます。 
character_matte_extractor


matte_shift : INT (default 0)

マットの膨張/収縮

プラス = 外側に広げる（欠け防止）

マイナス = 内側に縮める（白フチ削り） 
character_matte_extractor


min_foreground_area : INT (default 16)

小さすぎる前景の島をゴミとして除去します。 
character_matte_extractor


min_hole_area : INT (default 128)

キャラ内部にできた「背景色の穴」のうち、
面積が大きい穴だけを“背景として抜く”（透明にする）ためのしきい値です。
小さい穴（ハイライト等）は残しやすくなります。 
character_matte_extractor


OUTPUT

images : IMAGE

RGBA バッチ。キャラは不透明、背景はアルファ0。 
character_matte_extractor


5. Character Matte Extractor（CharacterMatteExtractor）
用途
DirectoryCharacterMatteExtractor の「単体画像（IMAGE入力）」版です。 
character_matte_extractor


パラメータは同一で、IMAGEバッチをそのまま処理して RGBA を返します。

6. Flat Color Posterizer（FlatColorPosterizer）
用途
入力画像から陰影・ハイライトをならし、指定した色数で **ベタ塗り化（ポスタライズ）**します。 
flat_color_posterizer


注意

opencv-python と scikit-learn が必要です。

INPUT（抜粋）

images : IMAGE

num_colors : INT (2–16, default 4)

smoothing_radius : INT (default 5)

edge_preserve : FLOAT (0–1, default 0.1)

color1_hex 〜 color8_hex（任意）

パレット色をHEXで指定（指定した分だけ優先して使います）。 
flat_color_posterizer


OUTPUT

images : IMAGE

サンプルワークフロー（透過.json）
透過.json は、線画抽出（単体／連番）とキャラ透過（連番）の使い方をまとめたサンプルです。 
透過


※ ワークフロー内に Fast Groups Muter (rgthree) が含まれるため、未導入環境ではそのノードだけ置き換え/削除してください。 
透過


使い始めのおすすめ設定
線画抽出（LineArt系）
threshold : 0.5 からスタート

線が薄く消える → 0.35〜0.45 に下げる

ゴミが多い／ベタが残りすぎ → 0.6 前後まで上げる

invert_input

白背景に黒線（一般的なスキャン線画）→ True 推奨

median_filter : 3

スキャン線画やアナログ原稿におすすめ

キャラ透過（CharacterMatte系）
まず background_color を「紙色」に合わせる（白紙以外はここが最重要）

次に threshold を調整

紙が残る → 少し上げる

キャラの薄い色まで抜ける → 少し下げる

紙が輪郭の内側に入り込む → close_gaps を上げる

白フチが気になる → matte_shift をマイナスへ（少しずつ）

欠けが気になる → matte_shift をプラスへ（少しずつ）

腕と体の隙間など「大きい穴」も抜きたい → min_hole_area を下げる
逆に目ハイライト等を残したい → min_hole_area を上げる

ライセンス
コード・ワークフローともにお好きに改造・利用して構いません。
作品内での使用時に作者表記は不要ですが、
もしクレジット頂ける場合は KALIN / comfyui-line-matte-extractor としていただけると嬉しいです。
