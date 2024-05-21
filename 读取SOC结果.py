import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
from math import sqrt


pred = np.load('D:\pythonProject\闲鱼\SOC\FWin/results/fwin_custom_ftMS_sl96_ll48_pl12_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_nw4_ws24_test_0/pred.npy') # 更换自己的路径
pred = pred[100:, 0 ,0].ravel() # 可以适当丢弃前100-400个值

print(pred.shape)

true = pd.read_csv('D:\pythonProject\闲鱼\SOC\FWin\dataset/Mixed_HWFET_25.csv') # 更换自己的路径
true = true['SOC'][-pred.shape[0]:,].ravel()


plt.plot(true,label='true')
plt.plot(pred,label='pred')
plt.legend()
plt.show()

plt.plot(true-pred)
plt.show()

# 评价指标
print("MAE:", mean_absolute_error(true,pred))
print("MSE:", mean_squared_error(true,pred))
print("RMSE:",sqrt(mean_squared_error(true,pred)))
print("R2: ", r2_score(true,pred))

