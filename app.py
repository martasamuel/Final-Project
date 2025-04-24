import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Preprocessing ----------
def load_image(img, max_size=512):
    if isinstance(img, str):
        image = Image.open(img).convert('RGB')
    else:
        image = img.convert('RGB')

    # Always resize to (max_size, max_size)
    in_transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = in_transform(image).unsqueeze(0)
    return image.to(device)

def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = torch.clamp(image, 0, 1)
    image = transforms.ToPILImage()(image)
    return image

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# ---------- VGG Feature Extraction ----------
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_features(image, model, layers=None):
    features = {}
    x = image
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
            if layers is None or name in layers:
                features[name] = x
    return features

# ---------- Streamlit UI ----------
st.title("🎨 Neural Style Transfer with Multiple Styles")
st.write("Upload a content image and apply styles blended from famous artists.")

uploaded_image = st.file_uploader("Upload your content image", type=["jpg", "png", "jpeg"])

# Style images in same directory
style_name_to_file = {
    "Leonardo": "leonardo.png",
    "Andy Warhol": "andy.png",
    "Picasso": "picasso.png",
    "Van Gogh": "van gogh.png"
}

selected_styles = st.multiselect("Select style(s):", options=list(style_name_to_file.keys()))

style_weights = {}
if selected_styles:
    st.write("Adjust style intensity for each:")
    for style in selected_styles:
        style_weights[style] = st.slider(f"{style} intensity", 0.0, 1.0, 0.5, 0.05)

color_weight = st.slider("🎨 Content color preservation (higher = more content color)", 0.0, 1.0, 1.0, 0.05)

if uploaded_image is not None and selected_styles:
    image = Image.open(uploaded_image)
    st.image(image, caption="Content Image", use_column_width=True)

    if st.button("✨ Run Style Transfer"):
        with st.spinner("Processing..."):

            content_img = load_image(image)
            content_features = get_features(content_img, vgg, content_layers)

            # Load and combine styles
            style_grams_combined = {layer: 0 for layer in style_layers}
            total_weight = sum(style_weights.values())

            for style_name in selected_styles:
                style_img = load_image(style_name_to_file[style_name])
                features = get_features(style_img, vgg, style_layers)
                for layer in style_layers:
                    gram = gram_matrix(features[layer])
                    style_grams_combined[layer] += gram * (style_weights[style_name] / total_weight)

            def run_style_transfer(content_img, content_features, style_grams, vgg, 
                                   content_weight=1e4, style_weight=1e2, num_steps=300):
                target = content_img.clone().requires_grad_(True).to(device)
                optimizer = torch.optim.Adam([target], lr=0.003)

                for step in range(1, num_steps+1):
                    target_features = get_features(target, vgg, style_layers + content_layers)

                    content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4']) ** 2)
                    style_loss = 0

                    for layer in style_layers:
                        target_gram = gram_matrix(target_features[layer])
                        style_gram = style_grams[layer]
                        _, d, h, w = target_features[layer].shape
                        style_loss += torch.mean((target_gram - style_gram) ** 2) / (d * h * w)

                    total_loss = (content_weight * color_weight) * content_loss + style_weight * style_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                return target

            output = run_style_transfer(
                content_img, content_features, style_grams_combined, vgg,
                content_weight=1e4, style_weight=1e2
            )
            result_image = tensor_to_image(output)
            st.image(result_image, caption="🎉 Stylized Output", use_column_width=True)
            result_image.save("output_image.png")
            st.success("Done! Saved as output_image.png")

else:
    st.info("📥 Please upload an image and select at least one style.")