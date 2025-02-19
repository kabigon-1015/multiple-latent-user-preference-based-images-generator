# multiple-latent-user-preference-based-images-generator

## feature_extraction

生成画像や入力画像の特徴抽出を行い、入力画像が GAN の潜在空間上でどの部分なのかを特定します。

## MCMC

各サンプリング手法のクラスが存在します。

## Projected GAN

https://github.com/autonomousvision/projected-gan から現在の Google Colab 環境で動作しない部分を改善したものになります。

## データセット

LSUN データセット内にある任意のカテゴリをデータセットとして取得するパイプラインを使用してください
下記のコマンドを実行することで jpg 形式でデータセットを作成することができます
image db path はカテゴリ名で[category_indices.txt](https://github.com/kabigon-1015/multiple-latent-user-preference-based-images-generator/blob/main/lsun/category_indices.txt "category_indices")に記載されているカテゴリ名を指定してください

```
python lsun/data.py export <image db path> --out_dir <output directory>
python lsun/data.py export *_val_lmdb --out_dir data
```

## Google Colab での実行例

```
!python estimate.py --weights_path '/content/drive/MyDrive/weights/resnet50_weights.pth' \
 --network_pkl 'https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/bedroom.pkl' \
 --output_dir '/content/drive/MyDrive/output_proposed' \
 --input_images '/path/to/input1.jpg' '/path/to/input2.jpg' '/path/to/input3.jpg'
```
