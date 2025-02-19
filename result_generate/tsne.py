import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(1, outputs_with_targets.shape[0] - 1), n_iter=5000, learning_rate=200)
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
        plt.scatter(outputs_2d_normalized[(target_index + 1) % 4, 0],
                    outputs_2d_normalized[(target_index + 1) % 4, 1],
                    c=color, s=200, marker='*', label=f'Target Image {i+1}')

        plt.annotate(f'Target {i+1}', (outputs_2d_normalized[(target_index + 1) % 4, 0], outputs_2d_normalized[(target_index + 1) % 4, 1]),
                    xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.legend()
    plt.colorbar(label='Similarity')
    plt.title('t-SNE Visualization of Encoded Images')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.xlim(-0.5, 1)
    plt.ylim(-0.5, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    plt.show()

# 使用例
# input_image_paths = [
#     "/content/drive/MyDrive/data/93.jpg",
#     "/content/drive/MyDrive/data/90.jpg",
#     "/content/drive/MyDrive/data/44.jpg"
# ]
# tsne_map(images, model, device, input_image_paths)
