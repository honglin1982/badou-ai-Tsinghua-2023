import scipy.special

# 初始化
i1, i2 = 0.05, 0.10
w1, w2, w3, w4, b1 = 0.15, 0.20, 0.25, 0.30, 0.35
w5, w6, w7, w8, b2 = 0.40, 0.45, 0.50, 0.55, 0.60
t1, t2 = 0.01, 0.99

# 前向传播
print('------前向传播----')
zh1 = w1 * i1 + w2 * i2 + b1 * 1
print(f'zh1 = {zh1}')
zh2 = w3 * i1 + w4 * i2 + b1 * 1
print(f'zh2 = {zh2}')

ah1 = scipy.special.expit(zh1)
print(f'ah1 = {ah1}')
ah2 = scipy.special.expit(zh2)
print(f'ah2 = {ah2}')

zo1 = w5 * ah1 + w6 * ah2 + b2 * 1
print(f'zo1 = {zo1}')
zo2 = w7 * ah1 + w8 * ah2 + b2 * 1
print(f'zo2 = {zo2}')

ao1 = scipy.special.expit(zo1)
print(f'ao1 = {ao1}')
ao2 = scipy.special.expit(zo2)
print(f'ao2 = {ao2}')

# 反向传播
print('------反向传播----')
Eo1 = (t1 - ao1) ** 2 / 2
print(f'Eo1 = {Eo1}')
Eo2 = (t2 - ao2) ** 2 / 2
print(f'Eo2 = {Eo2}')
Etotal = Eo1 + Eo2
print(f'Etotal = {Etotal}')

dw5 = (ao1 - t1) * (ao1 * (1 - ao1)) * ah1
print(f'dw5 = {dw5}')
w5_1 = w5 - dw5 * 0.5
print(f'w5+ = {w5_1}')

dw6 = (ao1 - t1) * (ao1 * (1 - ao1)) * ah2
print(f'dw6 = {dw6}')
w6_1 = w6 - dw6 * 0.5
print(f'w6+ = {w6_1}')

dw7 = (ao2 - t2) * (ao2 * (1 - ao2)) * ah1
print(f'dw7 = {dw7}')
w7_1 = w7 - dw7 * 0.5
print(f'w7+ = {w7_1}')

dw8 = (ao2 - t2) * (ao2 * (1 - ao2)) * ah2
print(f'dw8 = {dw8}')
w8_1 = w8 - dw8 * 0.5
print(f'w8+ = {w8_1}')

dw1 = ((ao1 - t1) * ao1 * (1 - ao1) * w5 + (ao2 - t2) * ao2 * (1 - ao2) * w7) * ah1 * (1 - ah1) * i1
print(f'dw1 = {dw1}')
w1_1 = w1 - dw1 * 0.5
print(f'w1+ = {w1_1}')

dw2 = ((ao1 - t1) * ao1 * (1 - ao1) * w5 + (ao2 - t2) * ao2 * (1 - ao2) * w7) * ah1 * (1 - ah1) * i2
print(f'dw2 = {dw2}')
w2_1 = w2 - dw2 * 0.5
print(f'w2+ = {w2_1}')

dw3 = ((ao1 - t1) * ao1 * (1 - ao1) * w6 + (ao2 - t2) * ao2 * (1 - ao2) * w8) * ah2 * (1 - ah2) * i1
print(f'dw3 = {dw3}')
w3_1 = w3 - dw3 * 0.5
print(f'w3+ = {w3_1}')

dw4 = ((ao1 - t1) * ao1 * (1 - ao1) * w6 + (ao2 - t2) * ao2 * (1 - ao2) * w8) * ah2 * (1 - ah2) * i2
print(f'dw4 = {dw4}')
w4_1 = w4 - dw4 * 0.5
print(f'w4+ = {w4_1}')
