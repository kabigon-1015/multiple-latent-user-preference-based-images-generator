import argparse
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_model(model_path, num_classes, device):
    model = torchvision.models.resnet50(pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_and_process_images(pipe, model, init_image, num_images, prompt, device, batch_size=4):
    outputs = []
    for i in tqdm(range(0, num_images, batch_size), desc="Generating and processing images"):
        batch = pipe(prompt=[prompt] * min(batch_size, num_images - i),
                     image=[init_image] * min(batch_size, num_images - i),
                     num_inference_steps=50,
                     strength=0.75,
                     guidance_scale=7.5).images

        processed_batch = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            output = model(processed_batch)

        outputs.append(output.cpu())
        torch.cuda.empty_cache()

    return torch.cat(outputs, dim=0)

def normalize(x):
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

def visualize_tsne(outputs_with_targets, target_indices, num_images):
    if isinstance(outputs_with_targets, torch.Tensor):
        outputs_with_targets = outputs_with_targets.cpu().detach().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, outputs_with_targets.shape[0] - 1))
    outputs_2d = tsne.fit_transform(outputs_with_targets)
    outputs_2d_normalized = normalize(outputs_2d)

    plt.figure(figsize=(12, 10))

    colors = ['red', 'green', 'yellow']
    for i, target_index in enumerate(target_indices):
        color = colors[i % len(colors)]
        plt.scatter(outputs_2d_normalized[target_index, 0],
                    outputs_2d_normalized[target_index, 1],
                    c=color, s=200, marker='*', label=f'Target Image {i+1}')

        start = i * num_images
        end = start + num_images if i < len(target_indices) - 1 else outputs_with_targets.shape[0] - len(target_indices)
        plt.scatter(outputs_2d_normalized[start:end, 0],
                    outputs_2d_normalized[start:end, 1],
                    c=color, alpha=0.5, label=f'Generated Images {i+1}' if i == 0 else "")

        plt.annotate(f'Target {i+1}', (outputs_2d_normalized[target_index, 0], outputs_2d_normalized[target_index, 1]),
                    xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.legend()
    plt.title('t-SNE Visualization of Encoded Images')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)

    global preprocess
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_model(args.resnet_weights, args.num_classes, device)

    all_outputs = []
    target_outputs = []

    for target_path in args.input_images:
        init_image = Image.open(target_path).convert("RGB").resize((512, 512))
        outputs = generate_and_process_images(pipe, model, init_image, args.num_images, args.prompt, device)
        all_outputs.append(outputs)

        target_image = Image.open(target_path).convert('RGB')
        target_tensor = preprocess(target_image).unsqueeze(0).to(device)
        with torch.no_grad():
            target_output = model(target_tensor)
        target_outputs.append(target_output.cpu())
        torch.cuda.empty_cache()

    all_outputs = torch.cat(all_outputs, dim=0)
    target_outputs = torch.cat(target_outputs, dim=0)
    outputs_with_targets = torch.cat([all_outputs, target_outputs], dim=0)

    target_indices = list(range(all_outputs.size(0), outputs_with_targets.size(0)))

    print("Processing complete.")
    print(f"Shape of outputs_with_targets: {outputs_with_targets.shape}")
    print(f"Target indices: {target_indices}")

    visualize_tsne(outputs_with_targets, target_indices, args.num_images)

# python script_name.py --resnet_weights /path/to/resnet_weights.pth --input_images /path/to/image1.jpg /path/to/image2.jpg /path/to/image3.jpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize images using Stable Diffusion and ResNet")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4", help="Stable Diffusion model ID")
    parser.add_argument("--resnet_weights", type=str, required=True, help="Path to ResNet weights")
    parser.add_argument("--input_images", nargs="+", required=True, help="Paths to input images")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate per input")
    parser.add_argument("--prompt", type=str, default="bedroom", help="Prompt for image generation")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes for ResNet")

    args = parser.parse_args()
    main(args)
