import argparse
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import dnnlib
import legacy
from sklearn.metrics.cosine_similarity import cosine_similarity

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

def find_most_similar_image(input_features, all_features):
    similarities = cosine_similarity(input_features.reshape(1, -1), all_features)
    most_similar_index = similarities.argmax()
    return most_similar_index

def create_image_pair(input_image_path, most_similar_image_path, output_path):
    input_image = Image.open(input_image_path)
    most_similar_image = Image.open(most_similar_image_path)
    
    # Resize images if they have different sizes
    size = (256, 256)  # You can adjust this size
    input_image = input_image.resize(size)
    most_similar_image = most_similar_image.resize(size)
    
    # Create a new image with both input and most similar side by side
    pair_image = Image.new('RGB', (size[0] * 2, size[1]))
    pair_image.paste(input_image, (0, 0))
    pair_image.paste(most_similar_image, (size[0], 0))
    
    pair_image.save(output_path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the classifier
    classifier = load_classifier(args.weights_path, args.num_classes, device)
    
    # Load the GAN model (if needed)
    gan_model = load_gan_model(args.network_pkl, device)
    
    # Load all feature vectors
    all_features = np.load(os.path.join(args.output_dir, "all_features.npy"))
    
    # Prepare image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process each input image
    for input_image_path in args.input_images:
        # Load and preprocess the input image
        input_image = Image.open(input_image_path).convert('RGB')
        input_tensor = transform(input_image)
        
        # Extract features from the input image
        input_features = extract_features(classifier, input_tensor, device)
        
        # Find the most similar image
        most_similar_index = find_most_similar_image(input_features, all_features)
        
        # Get the path of the most similar image
        most_similar_image_path = os.path.join(args.output_dir, f"{most_similar_index}.jpg")
        
        # Create and save the image pair
        output_pair_path = os.path.join(args.output_dir, f"pair_{os.path.basename(input_image_path)}")
        create_image_pair(input_image_path, most_similar_image_path, output_pair_path)
        
        print(f"Created image pair for {input_image_path}: {output_pair_path}")

# python create_input_image_pair.py --weights_path '/path/to/classifier_weights.pth' \
                # --network_pkl '/path/to/network-snapshot.pkl' \
                # --output_dir '/path/to/output/' \
                # --input_images '/path/to/input1.jpg' '/path/to/input2.jpg' '/path/to/input3.jpg'
# outputにall_features.npyファイルが存在することが前提
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image pairs with most similar generated images.")
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the classifier weights')
    parser.add_argument('--network_pkl', type=str, required=True, help='Path to the GAN network pickle file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing generated images and features')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes for the classifier')
    parser.add_argument('--input_images', nargs='+', required=True, help='Paths to input images')
    
    args = parser.parse_args()
    main(args)
