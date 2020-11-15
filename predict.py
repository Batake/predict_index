from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import math
import warnings
warnings.simplefilter('ignore')

# 探索する係数範囲の最小値, 最大値, 探索の試行回数
min_A, min_B, min_tc, min_M = -1000, -1000, 85.5, 0
max_A, max_B, max_tc, max_M = 1000, 1000, 95.5, 10
n_try = 10000

# 探索される対象の関数
def nonlinear_simple(x, a, b, c, d):
    return a + b*(abs(c-x)**d)

# 誤差の二乗の合計
def error_root_mean_squared(x, y, p_opt):
    y_predict = [nonlinear_simple(x_, p_opt[0], p_opt[1], p_opt[2], p_opt[3]) for x_ in x]
    error = np.mean([(y_ - y_m)**2 for y_, y_m in zip(y, y_predict)])
    return error

day_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
day_sum = [0] + [sum(day_month[0:(i+1)]) for i in range(11)] #その月までの合計日数: 1月なら0日, 2月までなら0+31=31, 3月までなら0+31+28=59...

# データのロードと整理
indice = pd.read_csv('./indice.csv', header=0, usecols=[1, 2, 3, 4]) #csvの1~4列目読み込み
for i, d_m in enumerate(day_month):
    indice["Month"][indice["Month"]-1==i] = day_sum[i] #月をその月までの合計日数に変換
indice['Date_Year'] = indice["Year"] + (indice["Month"] + indice["Day"])/365 # 年, 月, 日をそれまでの合算して年に変換
indice['Date_Year'] = indice['Date_Year'] - 1900 #上二桁を取り除く
x = indice["Date_Year"].values
y = indice["Open"].values

plt.plot(x, y)

# 87.6年以降の不要なデータの削除
tmp = x
tmp[tmp>87.6] = 0
y = y[tmp!=0]
x = x[tmp!=0]

# curve_fit関数で探索した係数(係数はa, b, ...の順にp_optへ入る)
p_opt_mine, _ = curve_fit(nonlinear_simple, x, y, maxfev=n_try, bounds=((min_A, min_B, min_tc, min_M), (max_A, max_B, max_tc, max_M)))

# 本での係数
p_opt_book = [327, -79, 87.65, 0.7]

# 暴落日の計算
print(p_opt_mine)
fall_year = p_opt_mine[2]
year = int(fall_year)
month = int((fall_year - year)*12) + 1
day = int(((fall_year-year)*12 - (month - 1))*30)
print("暴落日: " + str(1900 + year)+"年"+str(month)+"月"+str(day)+"日")

# エラー算出
print("自分の方法での誤差: ", error_root_mean_squared(x, y, p_opt_mine))
print("本での方法での誤差: ", error_root_mean_squared(x, y, p_opt_book))

# # 自分, 本のパラメータで算出したグラフ用データ
x_long = np.linspace(85.5, 90, 46)
y_long = [nonlinear_simple(x_l, p_opt_mine[0], p_opt_mine[1], p_opt_mine[2], p_opt_mine[3]) for x_l in x_long]
y_long_b = [nonlinear_simple(x_l, p_opt_book[0], p_opt_book[1], p_opt_book[2], p_opt_book[3]) for x_l in x_long]

# グラフ描画
plt.plot(x_long, y_long)
plt.plot(x_long, y_long_b)
plt.vlines([p_opt_mine[2]], 140, 360, "orange", linestyles='dashed') 
plt.vlines([p_opt_book[2]], 140, 360, "green", linestyles='dashed') 
plt.legend(["Raw", "Mine", "Book"])
plt.xlabel("Year")
plt.ylabel("S&P")
plt.ylim([160, 340])
plt.show()