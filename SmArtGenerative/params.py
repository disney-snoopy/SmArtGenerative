import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# model paths for streamlit
# need to be updated for gcp
vgg_model_path = '/home/byungjae/code/disney-snoopy/SmArtGenerative/pretrained_models/vgg16_pretrained'
segmentation_model_path = '/home/byungjae/code/disney-snoopy/SmArtGenerative/pretrained_models/torch_segmentation_finetuned'
