import numpy as np
from PIL import Image, ImageDraw
import os

# 打开原始图像
patch_number=1
def block_index_to_coordinates(block_index):
    if not 0 <= block_index < 64:
        raise ValueError("块索引必须在0到63之间")
    # 每个块的大小
    block_size = 16
    
    # 计算块的行号和列号
    row_index = block_index // 8  # 8个块为一行
    col_index = block_index % 8   # 8个块为一列
    
    # 计算块的左上角坐标
    top_left_x = col_index * block_size
    top_left_y = row_index * block_size
    
    return (top_left_x, top_left_y)

# 加载注意力映射
file_path = './data2/all_attention_maps.npy'
attention_map = np.load(file_path, allow_pickle=True)
print(attention_map.shape)
# 创建保存图像的文件夹（如果不存在）
save_img_dir = 'save_img2'
os.makedirs(save_img_dir, exist_ok=True)

patch_number = 1
s = 0
for batch_attention_maps in attention_map:
    # 计算每个块的列和
    column_sums = np.sum(batch_attention_maps, axis=2)
    min_weight_columns = np.argpartition(column_sums, patch_number, axis=2)[:, :, :patch_number]
    min_weight_columns = min_weight_columns.reshape(batch_attention_maps.shape[0], -1)

    # 遍历每个注意力映射
    for i, col_index_all in enumerate(min_weight_columns):
        mask = Image.new('L', (128, 128), 0)
        draw = ImageDraw.Draw(mask)
        for col_index in col_index_all:
            x, y = block_index_to_coordinates(col_index)
            block_size = 16
            x_end = x + block_size
            y_end = y + block_size
            draw.rectangle([x, y, x_end, y_end], fill=255)

        mask.save(f'{save_img_dir}/{s}_mask.png')
        s += 1