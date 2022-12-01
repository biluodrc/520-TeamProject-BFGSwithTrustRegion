import numpy as np

class TargetFunction:
    def __init__(self, func_idx):
        self.__func_idx = func_idx

    def func(self, x1, x2):
        if self.__func_idx == 1:
            return self.__func_1(x1, x2)
        elif self.__func_idx == 2:
            return self.__func_2(x1, x2)
        elif self.__func_idx == 3:
            return self.__func_3(x1, x2)

    def grad(self, x1, x2):
        if self.__func_idx == 1:
            return self.__grad_1(x1, x2)
        elif self.__func_idx == 2:
            return self.__grad_2(x1, x2)
        elif self.__func_idx == 3:
            return self.__grad_3(x1, x2)

    def toString(self):
        if self.__func_idx == 1:
            return "f(x1,x2) = exp^(-2*x1^2 - 5*x2^2)"
        elif self.__func_idx == 2:
            return "f(x1,x2) = 3*x1^2 + 5*x2^2 + 6*x1 - 7*x2 + 4"

    # The target function
    def __func_1(self, x1, x2):
        funcVal = -np.exp(-2 * x1 ** 2 - 5 * x2 ** 2)
        return funcVal
    # 1st derivative of target function
    def __grad_1(self, x1, x2):
        gradVal = np.array([[4 * x1 * np.exp(-2 * x1 ** 2 - 5 * x2 ** 2)],
                            [10 * x2 * np.exp(-2 * x1 ** 2 - 5 * x2 ** 2)]])
        return gradVal

    def __func_2(self, x1, x2):
        funcVal = 3 * x1 ** 2 + 5 * x2 ** 2 + 6 * x1 - 7 * x2 + 4
        return funcVal
    def __grad_2(self, x1, x2):
        gradVal = np.array([[6 * x1 + 6], [10 * x2 - 7]])
        return gradVal
