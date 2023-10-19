import argparse
import os
import json
import torch
from torchvision import models, transforms
from PIL import Image


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    processed_image = preprocess(image)
    return processed_image


def predict(image_path, model, topk, category_names, use_gpu):
    model.eval()
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    processed_image = process_image(image_path)
    processed_image = processed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(processed_image)

    probabilities, indices = torch.topk(output, topk)

    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]

    class_names = [category_names[class_id] for class_id in classes]

    # Move the probabilities tensor to CPU before converting to NumPy
    return probabilities.exp().cpu().numpy()[0], class_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")
    parser.add_argument('input', type=str, help="Path to the input image")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint")
    parser.add_argument('--top_k', type=int, default=1, help="Return the top K most likely classes")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help="Path to a JSON file mapping category names")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input image file not found: {args.input}")
        exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint file not found: {args.checkpoint}")
        exit(1)

    if not os.path.exists(args.category_names):
        print(f"Error: Category names file not found: {args.category_names}")
        exit(1)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    probabilities, class_names = predict(args.input, model, args.top_k, cat_to_name, args.gpu)

    for i in range(len(probabilities)):
        print(f"Prediction {i + 1}: {class_names[i]} with probability {probabilities[i]:.4f}")
