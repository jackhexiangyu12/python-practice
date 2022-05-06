n1=int(input())

def _int_iter():
    """根据回文数的定义。首先生成一个从0开始的整数无限序列"""
    global n1
    n=n1
    while True:
        yield n
        n -= 1


def _is_palindrome(n):
    """判断n是否为回文数，是就返回Ture，否就返回False"""
    L1 = list(str(n))
    L2 = L1[:]  # 利用列表的切片将L1复制出一个副本，并将副本赋值给L2（以免对L2进行操作时，改变L1）
    L2.reverse()  # 反转L2（reverse函数只对原Iterable进行反转，不会返回值）
    return L1 == L2


def palindromes():
    """利用filter进行筛选，只保留符合回文数要求的n值，并返回一个惰性的序列"""
    it = _int_iter()
    while True:
        n = next(it)
        yield n
        it = filter(_is_palindrome, it)


# 利用for循环，输出小于100000的所有回文数

# for num in palindromes():
#     if num < n1:
#         print(num)
#     else:
#         break

for num in range(n1,0,-1):
    if _is_palindrome(num):
        print(num)
        break