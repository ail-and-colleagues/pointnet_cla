# pointnet_cla
[trimesh](https://trimsh.org/index.html)で作成できるbox、cupsuleなどのプリミティブ形状を[pointnet](https://arxiv.org/abs/1612.00593)で分類するサンプル。

ゼミでのネタとして、そして建築学生含む情報分野が専門でない人向けに：
* データセットが簡単に用意できる
* 保守性はさておき読みやすい
* とりあえず動かしやすい

ように作成してみました。
二番目は個人の間隔により賛否異議あるかもですが、pythonを一通り学んで深層学習やってみようくらいですと、**合理的だけど知ってないと読めない実装で躓きやすいので比較的Pythonの初心者語で書いているつもり**です。

## install
必要なパッケージ（ライブラリ）はrequirements.txtにまとめてあるので、Terminalよりt次のコマンドを実行するとインストールされます。

```
pip install -r requirements.txt
```

必要に応じてCUDAやCuDNNもインストール・設定ください。GPUでくCPUで実行するならCUDAやCuDNNは不要です。

## contents
### trimesh_dataset_gen.py
[trimesh](https://trimsh.org/index.html)はPythonで利用できる三角形メッシュを操作するためのライブラリです。
このスクリプでは、trimeshで生成できるプリミティブな形状のうち、box・capsule・cylinder・scaled_sphereについて、その点群を作成し学習用のデータセットを作成します。なお、scaled_sphereについてはsphereをX軸・Y軸にランダムに拡大して作成してます。

mainのあたり（↓）にてdataset下のフォルダ名や、train・valのデータ数、点群の点数などを指定しています。辞書型をとるprimitivesがちょっとややこしいですが、keyに「"box"」のように意味、valueに「create_box」のように対応する点群を生成するための関数（のオブジェクト）が入っています。

```
if __name__ == "__main__":
    dataset_name = "trimesh_primitives"
    primitives = {
        "box": create_box,
        "capsule": create_capsule,
        "cylinder": create_cylinder, 
        "scaled_sphere": create_scaled_sphere}

    train_data_num, val_data_num = 100, 20
    point_num = 512
```

## links
