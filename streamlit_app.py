# Streamlit app: upload a bird photo to see the top-5 predicted species and a Grad-CAM heatmap.
# Needs app_config.json and classes.txt (exported by my notebook) in the same folder.
# The weights (final_model.pth) are too big for GitHub, so they are downloaded from Hugging Face.
# Run locally with:  streamlit run streamlit_app.py
import json
import os
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from torchvision import transforms

WEIGHTS_REPO = 'paing-chan/cub200-convnext-tiny'   # Hugging Face repo with final_model.pth

# Model name, image size and test results, all saved by the notebook
config = json.load(open('app_config.json'))
SIZE = config['image_size']
class_names = open('classes.txt').read().splitlines()

# Same preprocessing as the test images in the notebook: resize to 256, centre crop 224, normalise
crop = transforms.Compose([transforms.Resize(round(SIZE * 256 / 224)), transforms.CenterCrop(SIZE)])
preprocess = transforms.Compose([
    crop,
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Streamlit reruns the script on every click, so cache the model to load it only once
@st.cache_resource
def load_model():
    # use a local copy if there is one, otherwise download from Hugging Face
    path = 'final_model.pth' if os.path.exists('final_model.pth') else hf_hub_download(WEIGHTS_REPO, 'final_model.pth')
    model = timm.create_model(config['model'], pretrained=False, num_classes=200)   # my own weights are loaded below
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.eval()   # runs on CPU, Streamlit Cloud has no GPU


def overlay(img, cam):
    """Draws the Grad-CAM heatmap over the image (red = most important)."""
    cam = F.interpolate(cam[None, None], size=img.size[::-1], mode='bilinear')[0, 0].numpy()   # 7x7 -> image size
    heat = np.stack([np.clip(1.5 - np.abs(4 * cam - k), 0, 1) for k in (3, 2, 1)], axis=-1)   # blue -> green -> red
    blended = 0.55 * np.asarray(img) / 255 + 0.45 * heat
    return Image.fromarray(np.uint8(255 * blended))


def show_percent(p):
    """Formats a probability, showing '>99%' instead of 100%."""
    return '>99%' if p >= 0.995 else f'{p:.0%}'


def predict(model, image):
    """Returns the probabilities for all 200 species, the crop the model saw and the Grad-CAM heatmap."""
    x = preprocess(image).unsqueeze(0)   # add a batch dimension: (1, 3, 224, 224)

    # Run the model in two parts to keep the feature maps for Grad-CAM (so no torch.no_grad here)
    feats = model.forward_features(x)    # last conv stage: (1, 768, 7, 7)
    feats.retain_grad()                  # keep its gradient after backward()
    logits = model.forward_head(feats)   # pooling + classifier

    probs = logits.softmax(1)[0].detach()   # scores -> probabilities
    top = probs.argmax().item()

    # Grad-CAM: weight each feature map by its average gradient for the predicted species, keep the positive part
    model.zero_grad()
    logits[0, top].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(1))[0].detach()
    cam = cam / (cam.max() + 1e-8)                          # scale to 0..1
    seen = crop(image)
    return probs, seen, overlay(seen, cam)


# Example photos from the test set (two rows of four)
EXAMPLES = [('0000.jpg', 'Black footed Albatross'), ('0297.jpg', 'Horned Grebe'), ('0594.jpg', 'White Pelican'),
            ('0896.jpg', 'Black capped Vireo'), ('1199.jpg', 'Common Yellowthroat'), ('0096.jpg', 'Cardinal'),
            ('0344.jpg', 'Tricky: California Gull'), ('0711.jpg', 'Tricky: Fox Sparrow')]


@st.cache_data
def thumbnail(file):
    """Square crop so all thumbnails are the same size."""
    return ImageOps.fit(Image.open(f'examples/{file}'), (300, 300))


# CSS for a bigger, centred upload box with a dashed border and a hover highlight.
# It uses Streamlit's internal names, so the Streamlit version is pinned in requirements.txt.
CSS = """
<style>
[data-testid="stFileUploaderDropzone"] {
    min-height: 170px; flex-direction: column; justify-content: center; align-items: center; gap: 10px;
    border: 2px dashed rgba(128, 128, 128, 0.45); transition: border-color 0.2s, background-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1c83e1; background-color: rgba(28, 131, 225, 0.08); cursor: pointer;
}
[data-testid="stFileUploaderDropzone"]::before {content: 'Drag and drop a photo here'; font-weight: 600;}
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChip"])::before {content: 'Drop another photo to replace it';}
[data-testid="stFileUploaderDropzoneInstructions"] {text-align: center;}
</style>
"""

# ---------------- the page ----------------
st.set_page_config(page_title='Bird Species Classifier', page_icon='🐦', layout='wide')
st.markdown(CSS, unsafe_allow_html=True)
st.title('Bird Species Classifier')
st.write('Upload a photo of a bird and my model will guess which of the 200 species from the '
         'Caltech-UCSD Birds 200 dataset it is, and show where in the photo it looked.')

# Kept between reruns: which photo is shown ('upload' or an example path), and a key used to reset the upload box
st.session_state.setdefault('source', None)
st.session_state.setdefault('upload_key', 0)


def pick_example(path):
    """Shows the clicked example and clears the upload box."""
    st.session_state.source = path
    st.session_state.upload_key += 1   # a new key gives an empty upload box


# Input on the left, results on the right. height='stretch' keeps both boxes the same height.
left, right = st.columns(2, gap='medium')
with left, st.container(border=True, height='stretch'):
    uploaded = st.file_uploader('Your bird photo', type=['jpg', 'jpeg', 'png', 'webp'],
                                key=f'upload{st.session_state.upload_key}',
                                on_change=lambda: st.session_state.update(source='upload'))
    st.caption('Or try one of these photos from the test set:')
    thumbs = st.columns(4)
    for i, (file, name) in enumerate(EXAMPLES):
        path = f'examples/{file}'
        with thumbs[i % 4]:
            st.image(thumbnail(file), width='stretch')
            # the selected example gets a filled button
            st.button(name, key=file, width='stretch', on_click=pick_example, args=(path,),
                      type='primary' if st.session_state.source == path else 'secondary')

# The image to classify (None until a photo is uploaded or an example is picked)
image = None
if st.session_state.source == 'upload' and uploaded is not None:
    image = Image.open(uploaded).convert('RGB')
elif st.session_state.source not in (None, 'upload'):
    image = Image.open(st.session_state.source).convert('RGB')

with right, st.container(border=True, height='stretch'):
    if image is None:
        # placeholder message, centred in the box
        with st.container(height='stretch', vertical_alignment='center', horizontal_alignment='center'):
            st.markdown('#### No photo yet')
            st.caption('Upload a bird photo or pick one of the examples, and the prediction will appear here.')
    else:
        with st.spinner('Identifying the bird...'):   # the first run also downloads the weights
            probs, seen, heatmap = predict(load_model(), image)
        top5 = probs.topk(5)
        name, confidence = class_names[top5.indices[0]], top5.values[0].item()

        # Green if the model is more likely right than wrong (over 50%), amber if it is only guessing
        if confidence > 0.5:
            st.success(f'This looks like a **{name}** ({show_percent(confidence)} confident)')
        else:
            st.warning(f'Best guess: **{name}** ({show_percent(confidence)}). '
                       'Not very sure, so check the other options below too.')

        st.markdown('**Top-5 predictions**')
        for p, i in zip(top5.values.tolist(), top5.indices.tolist()):
            st.progress(p, text=f'{class_names[i]}: {show_percent(p)}')

        a, b = st.columns(2)   # crop and heatmap side by side
        a.image(seen, caption='What the model sees', width='stretch')
        b.image(heatmap, caption='Grad-CAM: red areas mattered most, blue least', width='stretch')

with st.expander('About the model'):
    st.markdown(f"- Model: `{config['model']}` (ConvNeXt-Tiny), pretrained on ImageNet-1k, then fine-tuned on the bird images at {SIZE}x{SIZE}px\n"
                '- Training data: all 4,829 training images of 200 bird species (settings were chosen on a 10% validation split first)\n'
                f"- Test Top-1 accuracy: **{config['test_top1']:.1%}**, average per-class accuracy: "
                f"**{config['test_per_class']:.1%}**, top-5 accuracy: **{config['test_top5']:.1%}**\n"
                '- Heatmap made with Grad-CAM on the last convolutional stage of the network')
