# Bird Species Classifier

A Streamlit web app that guesses which of 200 bird species is in a photo. It was made for my COS30082 Applied Machine Learning assignment.

**Try it here:** https://bird-classifier-paingchan.streamlit.app/

Upload a bird photo (or click one of the example photos) and the app shows:

- the species it thinks it is and how confident it is
- its top 5 guesses
- a Grad-CAM heatmap showing which part of the photo the model looked at

## The model

- **Model:** ConvNeXt-Tiny, pretrained on ImageNet-1k and then fine-tuned on bird photos
- **Dataset:** Caltech-UCSD Birds 200 (CUB-200), 200 species, 4,829 training images and 1,204 test images
- **Image size:** 224 x 224

Results on the test set:

| Metric | Score |
|---|---|
| Top-1 accuracy | 87.5% |
| Average per-class accuracy | 87.2% |
| Top-5 accuracy | 97.5% |

The model was trained in Google Colab. I compared it with ResNet50 and added one technique at a time (data augmentation, learning rate schedule, weight decay, dropout, MixUp/CutMix) to reduce overfitting. The full training process is in the notebook and the report for the assignment.

The trained weights (`final_model.pth`, about 112 MB) are too big for GitHub, so they are stored on Hugging Face instead:
https://huggingface.co/paing-chan/cub200-convnext-tiny

The app downloads them automatically the first time it starts.

## Files

| File | What it is |
|---|---|
| `streamlit_app.py` | The web app |
| `app_config.json` | Model name, image size and test results, exported from the training notebook |
| `classes.txt` | The 200 species names, in label order |
| `examples/` | Example photos from the test set |
| `requirements.txt` | Python packages the app needs |
| `.streamlit/config.toml` | Limits uploads to 10 MB |

## Running it on your own computer

You need Python 3.10 or newer.

```bash
git clone https://github.com/paingchan211/bird-classifier.git
cd bird-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On macOS or Linux, use `source venv/bin/activate` instead of `venv\Scripts\activate`.

The app then opens at http://localhost:8501. The first prediction takes a little longer because the model is downloaded from Hugging Face. If you already have `final_model.pth`, put it in this folder and the app will use it instead of downloading.

The app runs on the CPU, so no GPU is needed.
