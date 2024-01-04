import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions文章核心
def loss_coteaching(y_1, y_2, y_3, t, forget_rate, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda()#将loss_1中的元素从小到大排列,提取其在排列前对应的index(索引)输出
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    loss_3 = F.cross_entropy(y_3, t, reduce=False)
    ind_3_sorted = np.argsort(loss_3.data.cpu()).cuda()
    loss_3_sorted = loss_3[ind_3_sorted]

    remember_rate = 1 - forget_rate#R（T）
    num_remember = int(remember_rate * (len(loss_1_sorted)+len(loss_2_sorted)+len(loss_3_sorted))/3)

    pure_ratio_1 = np.sum(noise_or_not[ind_1_sorted[:num_remember].cpu()])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind_2_sorted[:num_remember].cpu()])/float(num_remember)
    pure_ratio_3 = np.sum(noise_or_not[ind_3_sorted[:num_remember].cpu()]) / float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    ind_3_update = ind_3_sorted[:num_remember]

    ind_3_update_new=torch.from_numpy(np.intersect1d(ind_1_update.cpu().numpy(),ind_2_update.cpu().numpy())).cuda()
    ind_2_update_new=torch.from_numpy(np.intersect1d(ind_3_update.cpu().numpy(),ind_1_update.cpu().numpy())).cuda()
    ind_1_update_new=torch.from_numpy(np.intersect1d(ind_2_update.cpu().numpy(),ind_3_update.cpu().numpy())).cuda()

    #ind_3_update_new = list(set(ind_1_update).union(set(ind_2_update)))
    #ind_2_update_new = list(set(ind_3_update).union(set(ind_1_update)))
    #ind_1_update_new = list(set(ind_2_update).union(set(ind_3_update)))

    ind_3_update_new2 = list(set(ind_1_update).intersection(set(ind_2_update)))
    ind_2_update_new2 = list(set(ind_3_update).intersection(set(ind_1_update)))
    ind_1_update_new2 = list(set(ind_2_update).intersection(set(ind_3_update)))

    #print(ind_1_update)
    #print(ind_2_update)
    #print(ind_3_update)
    #print(ind_1_update_new)
    #print(ind_2_update_new)
    #print(ind_3_update_new)
    #print("___________________________________")

    #ind_3_update_new = list(set(ind_1_update).intersection(set(ind_2_update)).union(set(ind_3_update)))
    #ind_2_update_new = list(set(ind_3_update).intersection(set(ind_1_update)).union(set(ind_2_update)))
    #ind_1_update_new = list(set(ind_2_update).intersection(set(ind_3_update)).union(set(ind_1_update)))

    loss_1_update = F.cross_entropy(y_1[ind_1_update_new],t[ind_1_update_new])
    loss_2_update = F.cross_entropy(y_2[ind_2_update_new], t[ind_2_update_new])
    loss_3_update = F.cross_entropy(y_3[ind_3_update_new], t[ind_3_update_new])

    # exchange选取最低的一定比率给另一个反向传播
    # loss_1_update = min(F.cross_entropy(y_1[ind_2_update], t[ind_2_update]),F.cross_entropy(y_1[ind_3_update], t[ind_3_update]))
    # loss_2_update = min(F.cross_entropy(y_2[ind_3_update], t[ind_3_update]),F.cross_entropy(y_2[ind_1_update], t[ind_1_update]))
    # loss_3_update = min(F.cross_entropy(y_3[ind_1_update], t[ind_1_update]),F.cross_entropy(y_3[ind_2_update], t[ind_2_update]))

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, torch.sum(loss_3_update)/num_remember,pure_ratio_1, pure_ratio_2,pure_ratio_3


