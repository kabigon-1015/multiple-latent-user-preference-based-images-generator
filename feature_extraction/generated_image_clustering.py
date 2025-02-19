import os
import argparse
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import cv2
import dnnlib
import legacy

# モデルのロード関数
def load_classifier(weights_path, num_classes, device):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()  # 評価モードに設定
    return model

# GANモデルのロード関数
def load_gan_model(network_pkl, device):
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    return G

# ディレクトリ作成関数
def create_output_dirs(base_dir, n_clusters):
    os.makedirs(base_dir, exist_ok=True)
    for i in range(n_clusters):
        os.makedirs(os.path.join(base_dir, str(i)), exist_ok=True)

# ランダムなone-hotベクトルを生成する関数
def one_hot_latent_vector(start_index, batch_size, z_dim):
    z_batch = torch.zeros(batch_size, z_dim)
    for i in range(batch_size):
        z_batch[i][(start_index + i) % z_dim] = 1.0  # ランダムなone-hotベクトル（例）
    return z_batch

# GAN画像生成とクラスタリング処理関数
def generate_and_classify_images(G, classifier, output_dir, n_clusters, batch_size, device):
    label = torch.zeros([1, G.c_dim], device=device)  # ラベル（必要に応じて変更）
    num_images = G.z_dim  # Gのz次元数を取得（生成する画像数）
    truncation_psi = 0.7  # トランケーションパラメータ（必要に応じて変更）
    noise_mode = 'const'  # ノイズモード（例: 'const', 'random'）

    # クラスタごとのディレクトリ作成
    create_output_dirs(output_dir, n_clusters)

    # 特徴量保存用ディレクトリ作成
    feature_dir = os.path.join(output_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)
    all_features = []  # すべての特徴ベクトルを格納するリスト

    global_count = 0
    for i in tqdm(range(0, num_images, batch_size), desc="Generating and processing images"):
        z_dim = G.z_dim  # Gのz次元数を取得
        z_batch = one_hot_latent_vector(i, batch_size, z_dim).to(device)

        # GANで画像生成
        img_batch = G(z_batch, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img_batch = (img_batch + 1) / 2 * 255  # [-1,1] -> [0,255]
        img_batch = img_batch.clamp(0, 255).to(torch.uint8)  # 型を uint8 に変換

        img_batch_input = img_batch.float() / 255.0  # 正規化して分類器に入力

        with torch.no_grad():
            outputs = classifier(img_batch_input.to(device))
            _, preds = torch.max(outputs, 1)

            for j, (pred, feature) in enumerate(zip(preds, outputs)):
                cluster_dir = os.path.join(output_dir, str(pred.cpu().numpy()))
                img_path = os.path.join(cluster_dir, f"{global_count}.jpg")

                # OpenCVで画像保存 (RGB -> BGR変換)
                cv2.imwrite(
                    img_path,
                    cv2.cvtColor(np.transpose(np.array(img_batch[j].cpu()), (1, 2, 0)), cv2.COLOR_RGB2BGR)
                )
                # 特徴ベクトルをリストに追加
                all_features.append(feature.cpu().numpy())
                global_count += 1

    # すべての特徴ベクトルを1つのnumpy配列に変換
    all_features_array = np.array(all_features)
    # 特徴ベクトルを1つのファイルとして保存
    np.save(os.path.join(output_dir, "all_features.npy"), all_features_array)

# メイン関数
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分類器のロード
    classifier = load_classifier(weights_path=args.weights_path,
                                  num_classes=args.num_classes,
                                  device=device)

    # GANモデルのロード
    gan_model = load_gan_model(network_pkl=args.network_pkl,
                               device=device)

    # GAN画像生成とクラスタリング処理
    generate_and_classify_images(G=gan_model,
                                  classifier=classifier,
                                  output_dir=args.output_dir,
                                  n_clusters=args.n_clusters,
                                  batch_size=args.batch_size,
                                  device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using GAN and classify them into clusters.")
    
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the pretrained weights of the classifier.')
    parser.add_argument('--network_pkl', type=str, required=True,
                        help='Path to the GAN network pickle file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the clustered images.')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes for the classifier.')
    parser.add_argument('--n_clusters', type=int, default=6,
                        help='Number of clusters for classification.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for image generation and classification.')

    args = parser.parse_args()
    main(args)
