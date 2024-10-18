# CaPaint
paper code
## inpantint生成数据步骤
### 在make_inpaint_data文件夹下，是生成npy训练数据的代码，包括从原始KTH数据如何生成inpaint之后的KTH数据
```python
cd make_inpaint_data
python batch_inpainting_KTH.py   #将掩码图像和原始图像经过图像修复模型重新生成改区域
python make_KTH_trainnpy.py     #将原始的数据保存为npy文件，用来训练
python make_KTH_testnpy.py      #将测试数据保存为npy文件,用来测试
python make_KTH_mask1npy.py     #将重新生成的图像保存为npy文件,用来训练
```
### 运行batch_inpainting_KTH.py 需要提前下载stable-diffusion-inpainting_KTH权重，和官方stable-diffusion-inpainting不同之处在于unet是微调过后的权重
1
