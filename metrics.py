import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import torch
# from torchmetrics.functional import structural_similarity_index_measure as ssim

def measurement_metrics(model, dataset):
    name = model + '_' + dataset
    name_with_mask = model + '_mask_' + dataset
    # 加载.npy文件
    data1 = np.load('/data01/dyf/CausaST3/output/'+name+'/Debug/results/Debug/sv/preds.npy') 
    print(data1.shape)
    data2 = np.load('/data01/dyf/CausaST3/output/'+name+'/Debug/results/Debug/sv/trues.npy') 
    print(data2.shape)

    preds = data1
    trues = data2

    maes = []
    mses = []
    ssims = []

    # 遍历每个图像
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # 计算每个图像的 MAE 和 MSE
            mae = np.mean(np.abs(preds[i, j] - trues[i, j]))
            mse = np.mean((preds[i, j] - trues[i, j])**2)
            # 计算每对图像的 SSIM
            # 计算每对图像的 SSIM
            min_dim = min(preds[i, j].shape[0], preds[i, j].shape[1])
            win_size = min(7, min_dim)
            if win_size % 2 == 0:
                win_size -= 1
            ssim_value = compare_ssim(preds[i, j], trues[i, j], multichannel=True, win_size=win_size, channel_axis=2, data_range=1.0)
            
            # 将结果添加到列表中
            maes.append(mae)
            mses.append(mse)
            ssims.append(ssim_value)
            #print(f"图像 [{i},{j}] 的 MAE: {mae}, MSE: {mse}")

    # 如果你想查看所有图像的平均 MAE 和 MSE
    print("不加mask")
    mae_result1 = np.mean(maes)
    print("所有图像的平均 MAE:", np.mean(maes))
    mse_result1 = np.mean(mses)
    print("所有图像的平均 MSE:", np.mean(mses))
    # 计算所有图像对的平均 SSIM
    ssim_result1 = np.mean(ssims)
    print('所有图像对的平均 SSIM:', np.mean(ssims))



    data1 = np.load('/data01/dyf/CausaST3/output/'+name_with_mask+'/Debug/results/Debug/sv/preds.npy') 
    print(data1.shape)
    data2 = np.load('/data01/dyf/CausaST3/output/'+name_with_mask+'/Debug/results/Debug/sv/trues.npy') 
    print(data2.shape)

    preds = data1
    trues = data2

    maes = []
    mses = []
    ssims = []
    # 遍历每个图像
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # 计算每个图像的 MAE 和 MSE
            mae = np.mean(np.abs(preds[i, j] - trues[i, j]))
            mse = np.mean((preds[i, j] - trues[i, j])**2)
            # 计算每对图像的 SSIM
            min_dim = min(preds[i, j].shape[0], preds[i, j].shape[1])
            win_size = min(7, min_dim)
            if win_size % 2 == 0:
                win_size -= 1
            ssim_value = compare_ssim(preds[i, j], trues[i, j], multichannel=True, win_size=win_size, channel_axis=2, data_range=1.0)
            
            # 将结果添加到列表中
            maes.append(mae)
            mses.append(mse)
            ssims.append(ssim_value)
            #print(f"图像 [{i},{j}] 的 MAE: {mae}, MSE: {mse}")

    # 如果你想查看所有图像的平均 MAE 和 MSE
    print("加mask")
    mae_result2 = np.mean(maes)
    print("所有图像的平均 MAE:", np.mean(maes))
    mse_result2 = np.mean(mses)
    print("所有图像的平均 MSE:", np.mean(mses))
    # 计算所有图像对的平均 SSIM
    ssim_result2 = np.mean(ssims)
    print('所有图像对的平均 SSIM:', np.mean(ssims))
    mae_rate = (mae_result1 - mae_result2)*100/mae_result1
    mse_rate = (mse_result1 - mse_result2)*100/mse_result1
    ssim_rate = (ssim_result2 - ssim_result1)*100/ssim_result2
    print('SSIM提升:', (ssim_result2 - ssim_result1)*100/ssim_result2)

    return mae_rate, mse_rate, ssim_rate, ssim_result2, ssim_result1