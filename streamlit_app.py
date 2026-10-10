# My Streamlit web app. You upload a bird photo and it shows the 5 species it thinks are most likely,
# plus a Grad-CAM heatmap of where the model looked.
# It needs app_config.json and classes.txt in the same folder (saved by the last cell of my notebook).
# The weights file (final_model.pth, 112 MB) is too big for GitHub, so it is stored in my free Hugging Face
# model repository and downloaded the first time the app starts.
# Run it on my own PC with:  streamlit run streamlit_app.py
import json
import os
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

WEIGHTS_REPO = 'paing-chan/cub200-convnext-tiny'   # my Hugging Face model repository with final_model.pth

# Settings saved by my notebook: the final model (ConvNeXt-Tiny), its image size (288px), the temperature fitted
# on the validation set (it fixes the under-confidence caused by label smoothing, so the percentages shown are
# honest) and its test results. Reading them from a file means the app always matches my last training run.
config = json.load(open('app_config.json'))
SIZE = config['image_size']
TEMPERATURE = config['temperature']
class_names = open('classes.txt').read().splitlines()

# Same resizing and normalising as the test images in my notebook, otherwise the predictions get worse.
# For 288px crops the short side is first resized to 329px (the same ratio as 224 / 256).
crop = transforms.Compose([transforms.Resize(round(SIZE * 256 / 224)), transforms.CenterCrop(SIZE)])
preprocess = transforms.Compose([
    crop,
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Streamlit reruns this whole script every time the user clicks something. @st.cache_resource makes it load the
# model only once and keep it in memory, otherwise every click would reload 112 MB of weights.
@st.cache_resource
def load_model():
    # use the local copy if I have one (on my PC), otherwise download it from Hugging Face (on Streamlit Cloud)
    path = 'final_model.pth' if os.path.exists('final_model.pth') else hf_hub_download(WEIGHTS_REPO, 'final_model.pth')
    # Same model as in training. pretrained=False because I load my own fine-tuned weights instead.
    # It has to be a CNN, because Grad-CAM needs the 2D feature maps of the last convolutional stage.
    model = timm.create_model(config['model'], pretrained=False, num_classes=200)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.eval()   # the free Streamlit server has no GPU, so everything runs on the CPU


def overlay(img, cam):
    """Lays the Grad-CAM heatmap (values 0 to 1) over the image the model saw, red = most important."""
    # upscale the small heatmap (9x9 for 288px images) to the image size so every pixel gets a value
    cam = F.interpolate(cam[None, None], size=img.size[::-1], mode='bilinear')[0, 0].numpy()
    # simple 'jet' colour map: low values blue, middle green, high values red
    heat = np.stack([np.clip(1.5 - np.abs(4 * cam - k), 0, 1) for k in (3, 2, 1)], axis=-1)
    blended = 0.55 * np.asarray(img) / 255 + 0.45 * heat   # 55% photo, 45% heatmap
    return Image.fromarray(np.uint8(255 * blended))


def show_percent(p):
    """Formats a probability for the user. Above 99.5% I show '>99%', because no model is ever truly 100% sure."""
    return '>99%' if p >= 0.995 else f'{p:.0%}'


def predict(model, image):
    """Returns the probabilities of all 200 species, the crop the model saw and the Grad-CAM heatmap."""
    x = preprocess(image).unsqueeze(0)   # the model expects a batch, so the shape becomes (1, 3, 288, 288)

    # I run the model in two halves instead of model(x), so I can get at the feature maps in between.
    # No torch.no_grad() here, because Grad-CAM needs gradients.
    feats = model.forward_features(x)    # output of the last conv stage: (1, 768, 9, 9)
    feats.retain_grad()                  # PyTorch normally throws away gradients of in-between tensors, keep this one
    logits = model.forward_head(feats)   # pooling + final layer, same result as model(x)

    # divide the raw scores by the temperature, then softmax turns them into probabilities.
    # This only changes how confident the model looks, the predicted species stays the same.
    probs = (logits / TEMPERATURE).softmax(1)[0].detach()
    top = probs.argmax().item()          # index of the most likely species

    # Grad-CAM, same method as my notebook: backpropagate the score of the predicted species to the feature maps.
    # The average gradient of each map says how important it is for this species, and the weighted sum of the
    # maps (only the positive part) shows which areas pushed the model towards its answer.
    model.zero_grad()
    logits[0, top].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)    # importance of each of the 768 channels
    cam = torch.relu((weights * feats).sum(1))[0].detach() # (9, 9) heatmap
    cam = cam / (cam.max() + 1e-8)                          # scale to 0..1 (1e-8 avoids dividing by 0)
    seen = crop(image)                                      # the 288x288 centre crop the model actually used
    return probs, seen, overlay(seen, cam)


# Example photos from the test set: five clear ones, plus two tricky ones where the model is unsure
# (it guesses Ring billed Gull for the California Gull, with the right answer second, and gets the Fox Sparrow
# right but with only about 40% confidence).
EXAMPLES = [('0000.jpg', 'Black footed Albatross'), ('0297.jpg', 'Horned Grebe'), ('0594.jpg', 'White Pelican'),
            ('0896.jpg', 'Black capped Vireo'), ('1199.jpg', 'Common Yellowthroat'),
            ('0344.jpg', 'Tricky: California Gull'), ('0711.jpg', 'Tricky: Fox Sparrow')]

# ---------------- the page ----------------
st.set_page_config(page_title='Bird Species Classifier', page_icon='🐦', layout='wide')
st.title('🐦 Bird Species Classifier')
st.write('Upload a photo of a bird and my model will guess which of the 200 species from the '
         'Caltech-UCSD Birds 200 dataset it is, and show where in the photo it looked.')

# st.session_state survives the reruns, so I use it to remember which photo is being shown:
# the uploaded one, or an example the user clicked.
if 'source' not in st.session_state:
    st.session_state.source = None

left, right = st.columns(2)   # two columns side by side: input on the left, results on the right
with left:
    uploaded = st.file_uploader('Bird photo', type=['jpg', 'jpeg', 'png', 'webp'],
                                on_change=lambda: st.session_state.update(source='upload'))
    st.caption('Or try one of these photos from the test set:')
    thumbs = st.columns(4)
    for i, (file, name) in enumerate(EXAMPLES):
        with thumbs[i % 4]:   # 4 thumbnails per row
            st.image(f'examples/{file}', width='stretch')
            if st.button(name, key=file, width='stretch'):
                st.session_state.source = f'examples/{file}'

# work out which image to use (None until the user uploads one or clicks an example)
image = None
if st.session_state.source == 'upload' and uploaded is not None:
    image = Image.open(uploaded).convert('RGB')
elif st.session_state.source and st.session_state.source != 'upload':
    image = Image.open(st.session_state.source).convert('RGB')

with right:
    if image is None:
        st.info('Upload a photo or click an example to see the prediction.')
    else:
        with st.spinner('Identifying the bird...'):   # first run also downloads the weights, so it can take a while
            probs, seen, heatmap = predict(load_model(), image)
        top5 = probs.topk(5)
        name, confidence = class_names[top5.indices[0]], top5.values[0].item()

        # The answer box: green when the model is confident, amber when it is only guessing.
        # Because of the temperature scaling the confidence is calibrated, so above 50% the model is more likely
        # right than wrong. Below that it is a guess, and I tell the user to look at the other options too.
        if confidence > 0.5:
            st.success(f'This looks like a **{name}** ({show_percent(confidence)} confident)', icon='✅')
        else:
            st.warning(f'Best guess: **{name}** ({show_percent(confidence)}). '
                       'Not very sure, so check the other options below too.', icon='⚠️')

        st.subheader('Top-5 predictions')
        for p, i in zip(top5.values.tolist(), top5.indices.tolist()):
            st.progress(p, text=f'{class_names[i]}: {show_percent(p)}')   # one bar per species

        a, b = st.columns(2)   # the crop and its heatmap next to each other, so they are easy to compare
        a.image(seen, caption='What the model sees', width='stretch')
        b.image(heatmap, caption='Grad-CAM: where it looked', width='stretch')
        st.caption('Red areas pushed the model most towards its answer, blue areas mattered least.')

with st.expander('About the model'):   # collapsed by default, opens when clicked
    st.markdown(f"- Model: `{config['model']}` (ConvNeXt-Tiny), pretrained, then fine-tuned on the bird images at {SIZE}x{SIZE}px\n"
                '- Training data: 4,829 images of 200 bird species\n'
                f"- Test Top-1 accuracy: **{config['test_top1']:.1%}**, average per-class accuracy: "
                f"**{config['test_per_class']:.1%}**, top-5 accuracy: **{config['test_top5']:.1%}**\n"
                f'- Confidence calibrated with temperature scaling (T = {TEMPERATURE}), so a shown 80% is right about 80% of the time\n'
                '- Heatmap made with Grad-CAM on the last convolutional stage of the network')
