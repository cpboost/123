# CaPaint
paper code

## Generate a mask by identifying environmental patches based on the attention map from self-supervised training.
batch_mask.py

## Steps for Generating Inpainting Data
### Running batch_inpainting_KTH.py requires downloading the ​Stable Diffusion Inpainting_KTH weights in advance. Unlike the official Stable Diffusion Inpainting, the ​UNet weights here have been fully fine-tuned specifically for different datasets. 
The fine-tuning process is implemented in the script fine_tune_unet.py.

### Under the make_inpaint_data folder, this is the code for generating .npy training data, including how to generate inpainting KTH data from the original KTH data.
```python
cd make_inpaint_data
python batch_inpainting_KTH.py   # Use the mask images and original images to regenerate the masked regions through the inpainting model
python make_KTH_trainnpy.py     # Save the original data as .npy files for training
python make_KTH_testnpy.py      # Save the test data as .npy files for testing
python make_KTH_mask1npy.py     # Save the regenerated images as .npy files for training
```

