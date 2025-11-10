# Muscle Segmentation

Our research focuses on MRI muscle segmentation, where we propose an automatic segmentation method for MRI scans. Manual muscle segmentation currently requires well-trained professionals and is extremely time-consuming. Our automatic segmentation method can help accurately and quickly segment target muscles for users who need the segmentation results. (This part will be shown in other Repo.) This research will benefit other researchers working in this area as well as hospitals that need efficient muscle analysis tools.

And in this Repo, we use Pytorch Unet as a baseline model to see how well can UNet segment the MRI Scans, including binary and multi-class muscle MRI images.

## Pytorch UNet

This repo is forked from https://github.com/milesial/Pytorch-UNet.


## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run training and testing.

## Data

The dataset is provided by The University of Sheffield. (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0299099)

Augmented lower limb MR images (MRI scans): https://orda.shef.ac.uk/articles/dataset/Augmented_lower_limb_MR_images/20440164

Augmented images associated segmentations (Annotated masks): https://orda.shef.ac.uk/articles/dataset/Augmented_images_associated_segmentations/20440203

We've processed the MRI Scans to images, 

 - binary dataset https://drive.google.com/file/d/1R0YO-PaGE4QbAqWtm-qX9X2jn_AYNyuj/view?usp=drive_link

 - 3-class dataset  https://drive.google.com/file/d/1jQvFTn8Jafam6kBkU3H-5qDhjDEt_7T0/view?usp=drive_link

- please contact me for more dataset we generated.




