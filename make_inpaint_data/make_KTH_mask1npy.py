import matplotlib.pyplot as plt
import numpy as np
import os

folder_path = './KTH_inpaint_mask1/'
# folder_names = [f for f in sorted(os.listdir(folder_path))]

def extract_number(folder_name):
    return int(''.join(filter(str.isdigit, folder_name)))
folder_names = sorted(os.listdir(folder_path), key=lambda x: extract_number(x))
print(folder_names[:10])

all_images_end = []
for i in folder_names:
    folder=i
    folder_path_i = os.path.join(folder_path, folder)
    print(folder_path_i)
    file_names = sorted([f for f in os.listdir(folder_path_i) if f.endswith('.png')], key=lambda x: int(x[:-4]))
    all_images = []

    for file_name in file_names:
        img_path = os.path.join(folder_path_i, file_name)
        rgb_image = plt.imread(img_path)
        all_images.append(rgb_image)
    all_images_np = np.stack(all_images,axis=0)
    all_images_end.append(all_images_np)
all_images_end = np.stack(all_images_end,axis=0)
all_images_end = all_images_end.transpose(0, 1, 4, 2, 3)

print(all_images_end.shape)
std = all_images_end.std()
mean = all_images_end.mean()
print(std)
print(mean)
new_data = (all_images_end-mean)/std
np.save('./KTH_train_mask1.npy', new_data)