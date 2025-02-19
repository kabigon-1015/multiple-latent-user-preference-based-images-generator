import argparse
import os
import numpy as np
import torch
from dataset_tool import make_transform
from mcmc.distribution import distribution
from mcmc.metropolis import Metropolis
from mcmc.replica_exchange_resampling_demc import reRDEMC
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import dnnlib
import legacy
from sklearn.metrics.cosine_similarity import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

translate = '0,0'
rotate = 0
truncation_psi = 1.0
noise_mode = 'const'

def est_latent(z,p):
  N = z.shape[0]
  D = z.shape[1]

  d = distribution(z,D,p)
  d.set_sigma(0.1*np.eye(D))

  mp = Metropolis(d,D)
  sample = mp.sampling()
  # DE = DE_MC(d,D)
  # DE_sample = DE.sampling()
  # re = Replica_exchange(d,D)
  # re_sample = re.sampling()
  reDE = reRDEMC(d,D)
  reDE_sample = reDE.sampling()

  # r = result_show2D(D)
  # r.estimate_result(sample, d.get_posterior(r.get_x_point_arr()))
  # r.estimate_result(reDE_sample, d.get_posterior(r.get_x_point_arr()))

  return sample, 1,1,reDE_sample

def load_classifier(weights_path, num_classes, device):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    return model

def load_gan_model(network_pkl, device):
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

def extract_features(model, image_tensor, device):
    with torch.no_grad():
        features = model(image_tensor.unsqueeze(0).to(device))
    return features.cpu().numpy().squeeze()

def find_most_similar_images(input_features, all_features, top_k=3):
    similarities = cosine_similarity(input_features.reshape(1, -1), all_features)
    top_indices = similarities.argsort()[0][-top_k:][::-1]
    return top_indices

def tsne_map(images, model, device, input_image_paths):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    torch.cuda.empty_cache()

    target_tensors = []
    for target_path in input_image_paths:
        target_image = Image.open(target_path).convert('RGB')
        target_tensor = preprocess(target_image).unsqueeze(0).to(device)
        target_tensors.append(target_tensor)

    all_target_tensors = torch.cat(target_tensors, dim=0).cpu()
    images_with_targets = torch.cat([images, all_target_tensors], dim=0)
    target_indices = list(range(images.size(0), images_with_targets.size(0)))

    outputs_with_targets = model(images_with_targets.to('cpu'))

    if isinstance(outputs_with_targets, torch.Tensor):
        outputs_with_targets = outputs_with_targets.cpu().detach().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, outputs_with_targets.shape[0] - 1), n_iter=5000, learning_rate=200)
    outputs_2d = tsne.fit_transform(outputs_with_targets)

    def normalize(x):
        return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

    outputs_2d_normalized = normalize(outputs_2d)

    plt.figure(figsize=(12, 10))

    plt.scatter(outputs_2d_normalized[:images.size(0), 0],
                outputs_2d_normalized[:images.size(0), 1],
                c='blue', alpha=0.5, label='Generated Images')

    colors = ['red', 'green', 'yellow', 'purple', 'orange']
    for i, target_index in enumerate(target_indices):
        color = colors[i % len(colors)]
        plt.scatter(outputs_2d_normalized[target_index, 0],
                    outputs_2d_normalized[target_index, 1],
                    c=color, s=200, marker='*', label=f'Target Image {i+1}')

        plt.annotate(f'Target {i+1}', (outputs_2d_normalized[target_index, 0], outputs_2d_normalized[target_index, 1]),
                    xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.legend()
    plt.colorbar(label='Similarity')
    plt.title('t-SNE Visualization of Encoded Images')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    plt.show()

def result_pgan(sample, G, device, input_image_paths, classifier, all_features):
    max_index = len(sample) - 1
    noise_index = 128
    index = [random.randint(0, max_index) for i in range(noise_index)]
    z = [sample[i][:] for i in index]
    z = np.array(z)
    label = torch.zeros([1, G.c_dim], device=device)

    z_batch = torch.zeros(noise_index, G.z_dim).to(device).float()

    # 入力画像ごとに最も似ている特徴量のインデックスを取得
    similar_indices = []
    for input_path in input_image_paths:
        input_image = Image.open(input_path).convert('RGB')
        input_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(input_image)
        input_features = extract_features(classifier, input_tensor, device)
        similar_indices.extend(find_most_similar_images(input_features, all_features, top_k=1))

    for i in range(noise_index):
        for j, idx in enumerate(similar_indices):
            z_batch[i, idx] = z[i][j]

    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    torch.cuda.empty_cache()

    z_batch = z_batch.cpu()
    img_batch = G(z_batch, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img_batch = (img_batch + 1) / 2
    img_batch = img_batch * 255
    img_batch = img_batch.clamp(0, 255).to(torch.uint8)

    tsne_map(img_batch.cpu(), classifier, device, input_image_paths)

    return img_batch.cpu()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = load_classifier(args.weights_path, args.num_classes, device)
    gan_model = load_gan_model(args.network_pkl, device)
    
    all_features = np.load(os.path.join(args.output_dir, "all_features.npy"))
    
    D = args.num_classes
    endD = args.num_classes
    iter = 1

    while D < endD + 1:
        print('Dim:', D)
        for i in range(iter):
            latent_space = []
            p = []
            print('step:', i+1)
            true_prefer = [4] * len(args.input_images) + [0] * (args.num_classes - len(args.input_images))
            for k in range(len(true_prefer)):
                z = [3.0 if i == k else 0 for i in range(D)]
                z = np.array(z)
                if true_prefer[k] > 0:
                    p.append(true_prefer[k])
                    latent_space.append(z)
            new_latent_space = np.array(latent_space)
            p = np.array(p)
            print(new_latent_space)
            print(p)
            sample, DE_sample, re_sample, reDE_sample = est_latent(new_latent_space, p)
            s = [sample, re_sample, DE_sample, reDE_sample]

            result_pgan(s[3], gan_model, device, args.input_images, classifier, all_features)

        iter = 1
        D += 1

# python estimate.py --weights_path '/path/to/classifier_weights.pth' \
#                       --network_pkl '/path/to/network-snapshot.pkl' \
#                       --output_dir '/path/to/output/' \
#                       --input_images '/path/to/input1.jpg' '/path/to/input2.jpg' '/path/to/input3.jpg'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize images based on input preferences.")
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the classifier weights')
    parser.add_argument('--network_pkl', type=str, required=True, help='Path to the GAN network pickle file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing generated images and features')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes for the classifier')
    parser.add_argument('--input_images', nargs='+', required=True, help='Paths to input images')
    
    args = parser.parse_args()
    main(args)
