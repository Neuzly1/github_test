import cv2
import os
import numpy as np


color = np.array([255])     # 只区分背景(pixel = 0)和目标对象(pixel = 255)，因此数组中只写入255.

def diceCoeffic(pred, gt, eps=1e-5):
    r""" computational formula：
        pred:input image pixel
        gt:ground truth
        dice = (2 * tp) / (2 * tp + fp + fn)
        return dice
    """
    dice = []
    # pre_all = []
    # re_all = []
    # f1_all = []
    N = gt.size
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)            # array整形
    for i in range(len(color)):
        tp = np.sum( (pred_flat == color[i]) * (gt_flat == color[i]) )  # 利用判别式返回0/1的性质，完成对目标像素值的校验.
        fp = np.sum( (pred_flat == color[i]) * (gt_flat == 0) )         #
        fn = np.sum( (pred_flat == 0) * (gt_flat == color[i]) )         #
        score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        # pre = tp/(tp+fp)
        # re = tp/(tp+fn)
        # f1 = 2*pre*re/(pre+re)
        # score=score.sum() / N
        dice.append(score)              # 向列表中添加元素
        # pre_all.append(pre)
        # re_all.append(re)
        # f1_all.append(f1)
    return dice


def save_txt(img_list, dice_list, save_path):
    """
    将每张分割结果的dice和训练集的平均dice存到txt文件中
    img_list:图片列表
    dice_list:dice列表
    save_path:存储路径
    """
    if os.path.isfile(save_path):
        os.remove(save_path)             # 用于删除指定路径的文件
    with open(save_path, "a") as f:      # 以附加形式打开文件，保证不会覆盖内容的前提下，进行写入操作
        for i in range( len(img_list) ):
            f.write('{}:{}\n'.format(img_list[i], dice_list[i]))   #将文本写入文件
        f.close()

    # method 2
    # with open(save_path, "w") as f:
    #     for i in range(len(img_list)):
    #         f.write('{}:{}\n'.format(img_list[i], dice_list[i]))
    #     f.close()


img_path = r'E:/Pycharm_workspace/experiment_3/head_dataset/test/predict'   # img的绝对路径，‘r’防止str转义
pred_list = []
gt_list = []

for a in os.listdir(img_path):                      # 遍历img_path下的所有文件
    '''得到pred、gt的路径值'''
    if a.find('pred') != -1:                        # a.find('pred')返回第一个与之匹配字符串的位置，否则返回-1
        pred_list.append(os.path.join(img_path, a)) # 路径拼接,得到pred图片的绝对路径
        b = a.split('_')[0]                         # 实际上就是通过字符串分割，获得所处理的'图组编号',结合数据集易知
        gt_list.append(os.path.join(img_path, b+'_label.png')) # 路径拼接,得到gt(即label)图片的绝对路径

pred_list.append('mean dice')
dice_list = []
print("zhang lanyi's result:")
for i in range( len(pred_list)-1 ):
    '''得到一些列dice值'''
    pred = cv2.imread(pred_list[i], 0)              # 以8位深度,单channel读入图片
    gt = cv2.imread(gt_list[i], 0)
    dice = diceCoeffic(pred, gt)                    # 计算每张pred的dice
    dice_list.append(dice)                          # 将dice值放入dice_list之中
    print(pred_list[i], dice)                       # 打印对应文件的的dice值

dice_all = np.array(dice_list)                      # 将list转换为array,节省内存
dice_mean_0 = np.mean(dice_all, axis=0)             # axis = 0,计算列均值
dice_mean_1 = np.mean(dice_all, axis=1)             # axis = 1,计算行均值
dice_mean_1 = dice_mean_1.reshape(-1, 1)            # 将dice_mean_1变成1列，行数自动计算
dice_and_mean = np.hstack( (dice_all, dice_mean_1) )  # 水平方向堆叠(拼接)数组
dice_r = np.mean(dice_and_mean, axis=0)             # axis = 0,计算列均值
dice_final = np.vstack((dice_and_mean, dice_r))     # 垂直方向堆叠(拼接)数组

txt_file = r'E:/Pycharm_workspace/experiment_3/head_dataset/train_dice.txt'
save_txt(pred_list, dice_final, txt_file)
