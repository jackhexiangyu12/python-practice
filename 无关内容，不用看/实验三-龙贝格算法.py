from math import sin, pi

import numpy as np


#
# def f(x):
#     return sin(x) / x
#
#
# a = 0
# b = 1
# eps = 1e-6
# k = 0
# h = b - a
# T = [[]]
# T[0][0] = h / 2 * (f(a) + f(b))
# # T[0][(b-a)/(pow(2,k))]=
#
# m = 0
# T[m][k] = (pow(4, m) / (pow(4, m) - 1)) * T[m - 1][h / 2] - (1 / (pow(4, m) - 1)) * T[m - 1][h]

eps=6

def func(x):
    return (x*x*x+sin(x))/x


class Romberg:
    def __init__(self, integ_downlimit, integ_uplimit):
        self.integ_downlimit = integ_downlimit
        self.integ_uplimit = integ_uplimit

    def calc(self):
        t_seq1 = np.zeros(9, 'f')
        s_seq2 = np.zeros(8, 'f')
        c_seq3 = np.zeros(7, 'f')
        r_seq4 = np.zeros(6, 'f')
        r_seq5 = np.zeros(5, 'f')
        r_seq6 = np.zeros(4, 'f')
        r_seq7 = np.zeros(3, 'f')
        r_seq8 = np.zeros(2, 'f')
        r_seq9 = np.zeros(1, 'f')

        hm = [(self.integ_uplimit - self.integ_downlimit) / 2 ** i for i in range(0, 9)]
        print(hm)

        fa = func(self.integ_downlimit)
        fb = func(self.integ_uplimit)

        t_seq1[0] = (1 / 2) * (self.integ_uplimit - self.integ_downlimit) * (fa + fb)

        for i in range(1, 9):
            sum = 0
            for j in range(1, 2 ** i, 2):  # 很奇怪，为什么不是等比而是等差，因为是递推公式的项,因为奇数项所以是步长是2
                sum = sum + func(self.integ_downlimit + j * hm[i])*hm[i]
            t_seq1[i] = sum + 1 / 2 * t_seq1[i - 1]
        print('T序列（第一列，即下标为0的序列）', t_seq1)
        # 为什么需要print('T序列：'+ str(list(t_seq1)))
        # round(x,n)返回x的小数点后n位的四舍五入值
        # s_seq2=[round((4*t_seq1[i+1]-t_seq1[i])/3,7)for i in range(0,4)]

        # 等价于以下代码

        for k, _ in enumerate(t_seq1):
            k=k-1
            s_seq2[k] = round((4 ** 1 * t_seq1[k + 1] - t_seq1[k]) / (4 ** 1 - 1), eps)
        print('S序列（第二列，即下标为1的序列）', s_seq2)
        for k, _ in enumerate(s_seq2):
            k=k-1
            c_seq3[k] = round((4 ** 2 * s_seq2[k + 1] - s_seq2[k]) / (4 ** 2 - 1), eps)
        print('C序列（第三列，即下标为2的序列）', c_seq3)
        for k, _ in enumerate(c_seq3):
            k=k-1
            r_seq4[k] = round((4 ** 3 * c_seq3[k + 1] - c_seq3[k]) / (4 ** 3 - 1), eps)
        print('R序列（第四列，即下标为3的序列）', r_seq4)
        for k, _ in enumerate(r_seq4):
            k=k-1
            r_seq5[k] = round((4 ** 4 * r_seq4[k + 1] - r_seq4[k]) / (4 ** 4 - 1), eps)
        print('R序列（第五列，即下标为4的序列）', r_seq5)
        for k, _ in enumerate(r_seq5):
            k=k-1
            r_seq6[k] = round((4 ** 5 * r_seq5[k + 1] - r_seq5[k]) / (4 ** 5 - 1), eps)
        print('R序列（第六列，即下标为5的序列）', r_seq6)
        for k, _ in enumerate(r_seq6):
            k=k-1
            r_seq7[k] = round((4 ** 6 * r_seq6[k + 1] - r_seq6[k]) / (4 ** 6 - 1), eps)
        print('R序列（第七列，即下标为6的序列）', r_seq7)
        for k, _ in enumerate(r_seq7):
            k=k-1
            r_seq8[k] = round((4 ** 7 * r_seq7[k + 1] - r_seq7[k]) / (4 ** 7 - 1), eps)
        print('R序列（第八列，即下标为7的序列）', r_seq8)


        return 'end'  # 为啥要返回end？




rom = Romberg(0.3, 0.8)
print(rom.calc())

