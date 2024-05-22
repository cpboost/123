import os
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import shutil
def load_image(image_path):
    return Image.open(image_path).convert("RGB")
def resize_image(image, target_size):
    return image.resize(target_size,Image.LANCZOS)
# 设置文件夹路径
img_dir = "KTH"
mask_dir = "mask_img"
output_dir = "KTH_inpant_mask1"

# 加载模型
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "../checkpoint/stable-diffusion-inpainting_KTH", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
all_mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
i= 0

def extract_number(folder_name):
    return int(''.join(filter(str.isdigit, folder_name)))
sorted_subfolders = sorted(os.listdir(img_dir), key=lambda x: extract_number(x))[:350]
# 遍历 img 文件夹内的子文件夹
for subfolder_index, subfolder in enumerate(sorted_subfolders):
    subfolder_path = os.path.join(img_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # 获取子文件夹内的前10个图片路径和后10个图片路径
        img_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
        first_10_files = img_files[:10]
        last_10_files = img_files[10:20] 
        print( first_10_files )
        mask_i = f'{str(i)}_mask.png'
        mask_path = os.path.join(mask_dir, mask_i)
        mask_image = load_image(mask_path)
        # 确保输出目录存在
        output_subfolder_path = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder_path, exist_ok=True)

        # 合成前10个图像
        for img_file in first_10_files:
            img_path = os.path.join(subfolder_path, img_file)

            init_image = load_image(img_path)

            # 使用 Stable Diffusion Inpainting Pipeline 合成图像
            generated_image = pipe(image=init_image, mask_image=mask_image, prompt="").images[0]
            # if generated_image.mode != "RGBA":
            #     generated_image = generated_image.convert("RGBA")
            resized_image = resize_image(generated_image, init_image.size)
            # 保存生成的图像，文件名和原始图片相同
            resized_image.save(os.path.join(output_subfolder_path, img_file),format='PNG')
        for img_file in last_10_files:
            src_path = os.path.join(subfolder_path, img_file)
            dst_path = os.path.join(output_subfolder_path, img_file)
            shutil.copy(src_path, dst_path)
        i+=1

