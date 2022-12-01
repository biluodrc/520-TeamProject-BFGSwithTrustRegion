# Implementation of BFGS

import random

import numpy as np
from matplotlib import pyplot as plt


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
        funcVal = -np.exp(-2 * x1**2 - 5 * x2**2)
        return funcVal

    # 1st derivative of target function
    def __grad_1(self, x1, x2):
        gradVal = np.array([[4 * x1 * np.exp(-2 * x1**2 - 5 * x2**2)],
                            [10 * x2 * np.exp(-2 * x1**2 - 5 * x2**2)]])
        return gradVal

    def __func_2(self, x1, x2):
        funcVal = 3 * x1**2 + 5 * x2**2 + 6 * x1 - 7 * x2 + 4
        return funcVal

    def __grad_2(self, x1, x2):
        gradVal = np.array([[6 * x1 + 6], [10 * x2 - 7]])
        return gradVal


class BFGS_With_Trust_Region(object):
    def __init__(self,
                 seed=None,
                 epsilon=1.e-4,
                 maxIter=300,
                 init_point=None,
                 func_idx=1):
        self.__seed = seed  # The random seed
        self.__epsilon = epsilon  # Accuracy
        self.__maxIter = maxIter  # The maximum number of iteration
        self.__init_point = init_point  # The beginning point of interation

        self.__xPath = list()  # Store the path of optimization points
        self.__fPath = list(
        )  # Store the path of values of the target function

        self.__delta_max = 1  # The maximum value of trust region radius
        self.__eta = 0.125  # Hyperparameter Eta to control the optimization

        self.__deltaList = list()  # Store the change of trust region radius
        self.__converge_flag = False  # If the target function is converged, set this as True

        self.__targetFunction = TargetFunction(func_idx)

    def solve(self):
        self.__init_path()

        xCurr = self.__get_init_point(
        )  # Initialize the first point of optimization
        fCurr = self.__targetFunction.func(
            xCurr[0, 0],
            xCurr[1,
                  0])  # Calculate the target function value of the first point
        gCurr = self.__targetFunction.grad(
            xCurr[0, 0],
            xCurr[1, 0])  # Calculate the derivative value of the first point
        BCurr = self.__init_B(
            xCurr.shape[0])  # Initialize the matrix D with Identity Matrix
        deltaCurr = 0.5  # trust region radius
        self.__save_path(xCurr, fCurr, deltaCurr)

        # Main body of optimization
        for i in range(self.__maxIter):
            if self.__is_converged(gCurr):
                self.__converge_flag = True
                self.__print_MSG()
                break

            # Implementation of BFGS
            dCurr = -np.matmul(BCurr,
                               gCurr)  # The current direction of optimization
            alpha = self.__calc_alpha_by_ArmijoRule(
                xCurr, fCurr, gCurr,
                dCurr)  # Calculate the alpha by the Armijo Rule
            s_k = alpha * dCurr
            flag = False
            if np.linalg.norm(s_k) > deltaCurr:
                s_k = s_k / np.linalg.norm(s_k) * deltaCurr
                flag = True

            # Implementation of Trust Region method
            m_xk_sk = self.__targetFunction.func(xCurr[0], xCurr[1]) + np.dot(
                s_k.T, self.__targetFunction.grad(xCurr[0, 0], xCurr[1, 0])
            ) + 1 / 2 * np.dot(np.dot(s_k.T, np.linalg.inv(BCurr)), s_k)
            m_xk = self.__targetFunction.func(xCurr[0], xCurr[1])
            rho = (self.__targetFunction.func(xCurr[0, 0], xCurr[1, 0]) -
                   self.__targetFunction.func(xCurr[0, 0] + s_k[0, 0],
                                              xCurr[1, 0] + s_k[1, 0])) / (
                                                  m_xk - m_xk_sk)
            # Update the radius of trust region
            deltaNext = -1
            if rho < 1 / 4:
                deltaNext = 1 / 4 * deltaCurr
            else:
                if flag == True or (rho > 3 / 4
                                    and np.linalg.norm(s_k) == deltaCurr):
                    deltaNext = min(2 * deltaCurr, self.__delta_max)
                else:
                    deltaNext = deltaCurr

            # Update the current point of optimization
            if rho > self.__eta:
                xNext = xCurr + s_k
            else:
                xNext = xCurr

            # Update the matrix B by BFGS algorithm
            fNext = self.__targetFunction.func(xNext[0, 0], xNext[1, 0])
            gNext = self.__targetFunction.grad(xNext[0, 0], xNext[1, 0])
            if rho > self.__eta:
                DNext = self.__update_B_by_BFGS(xCurr, gCurr, xNext, gNext,
                                                BCurr)
            else:
                DNext = BCurr

            xCurr, fCurr, gCurr, BCurr = xNext, fNext, gNext, DNext
            deltaCurr = deltaNext
            self.__save_path(xCurr, fCurr, deltaCurr)
        else:
            if self.__is_converged(gCurr):
                self.__print_MSG()
            else:
                print("BFGS_With_Trust_Region not converged after {} steps!".
                      format(self.__maxIter))

    def show(self):
        # Draw the process of the function optimization and the plot of how trust region changes.
        if not self.__xPath:
            self.solve()

        for i in range(len(self.__fPath)):
            fig = plt.figure(figsize=(10, 4))
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)

            # The left picture, about the change of target function value.
            ax1.plot(np.arange(i + 1), self.__fPath[0:i + 1], "k.")
            ax1.plot(0, self.__fPath[0], "go", label="starting point")
            if i == len(self.__fPath) - 1:
                ax1.plot(len(self.__fPath) - 1,
                         self.__fPath[-1],
                         "r*",
                         label="solution")
            ax1.set(xlabel="$iterCnt$", ylabel="$iterVal$")
            ax1.legend()

            # The right picture, about the line of optimization process.
            x1Path = list(item[0] for item in self.__xPath)
            x2Path = list(item[1] for item in self.__xPath)
            min_1 = min(x1Path)
            max_1 = max(x1Path)
            min_2 = min(x2Path)
            max_2 = max(x2Path)
            x1 = np.linspace(min_1 - 0.5, max_1 + 0.5, 50)
            x2 = np.linspace(min_2 - 0.5, max_2 + 0.5, 50)
            x1, x2 = np.meshgrid(x1, x2)
            f = self.__targetFunction.func(x1, x2)
            ax2.contour(x1, x2, f, levels=36)
            ax2.plot(x1Path[0:i + 1], x2Path[0:i + 1], "k--", lw=2)
            ax2.plot(x1Path[0], x2Path[0], "go", label="starting point")
            ax2.plot(x1Path[-1], x2Path[-1], "r*", label="solution")
            ax2.set(xlabel="$x_1$", ylabel="$x_2$")
            ax2.legend()

            # The right picture, about the change of trust region.
            deltaPath = self.__deltaList
            # for j in range(i):
            #     r = deltaPath[j]
            #     x1 = x1Path[j]
            #     x2 = x2Path[j]
            #     # (j*0.1+0.1, j*0.1+0.1, j*0.1+0.1, j*0.1+0.1)
            #     circle = plt.Circle((x1, x2), r, color=[round(random.uniform(0,1),1) for _ in range(3)], fill=True)
            #     plt.gcf().gca().add_artist(circle)
            r = deltaPath[i]
            x1 = x1Path[i]
            x2 = x2Path[i]
            # (j*0.1+0.1, j*0.1+0.1, j*0.1+0.1, j*0.1+0.1)
            circle = plt.Circle(
                (x1, x2),
                r,
                color=[round(random.uniform(0.5, 1), 1) for _ in range(3)],
                fill=True)
            plt.gcf().gca().add_artist(circle)

            fig.tight_layout()
            plt.savefig(str(i) + '.jpg')
            fig.show("bfgs.png")
            plt.pause(2)

    def is_converged(self):
        return self.__converge_flag

    def __print_MSG(self):
        # Print the related information, such as interation step, initial seed
        # and the final solution on the Terminal screen.
        print("Target function: " + self.__targetFunction.toString())
        print("Iteration steps: {}".format(len(self.__xPath) - 1))
        print("Seed: {}".format(self.__xPath[0].reshape(-1)))
        print("Solution: {}".format(self.__xPath[-1].reshape(-1)))
        print('-'*5 + 'TEST PASS' + '-'*5)

    def __is_converged(self, gCurr):
        # If the function is converged: return True
        # If the function is not converged: return False
        if np.linalg.norm(gCurr) <= self.__epsilon:
            return True
        return False

    def __update_B_by_BFGS(self, xCurr, gCurr, xNext, gNext, BCurr):
        # Update the matrix B based on BFGS algorithm
        sk = xNext - xCurr
        yk = gNext - gCurr
        rk = 1 / np.matmul(yk.T, sk)[0, 0]

        term1 = rk * np.matmul(sk, yk.T)
        term2 = rk * np.matmul(yk, sk.T)
        I = np.identity(term1.shape[0])
        term3 = np.matmul(I - term1, BCurr)
        term4 = np.matmul(term3, I - term2)
        term5 = rk * np.matmul(sk, sk.T)

        Bk = term4 + term5
        return Bk

    def __calc_alpha_by_ArmijoRule(self,
                                   xCurr,
                                   fCurr,
                                   gCurr,
                                   dCurr,
                                   c=1.e-4,
                                   v=0.5):
        # Calculate the learning rate alpha by using the Armijo Rule.
        i = 0
        alpha = v**i
        xNext = xCurr + alpha * dCurr
        fNext = self.__targetFunction.func(xNext[0, 0], xNext[1, 0])

        while True:
            if fNext <= fCurr + c * alpha * np.matmul(dCurr.T, gCurr)[0, 0]:
                break
            i += 1
            alpha = v**i
            xNext = xCurr + alpha * dCurr
            fNext = self.__targetFunction.func(xNext[0, 0], xNext[1, 0])

        return alpha

    def __init_B(self, n):
        # initialize the matrix B with an Identity Matrix
        B = np.identity(n)
        return B

    def __init_seed(self, seed):
        if seed is None:
            seed = np.random.uniform(-100, 100, 2)

        seed = np.array(seed).reshape((2, 1))
        return seed

    def __init_path(self):
        self.__xPath.clear()
        self.__fPath.clear()

    def __save_path(self, xCurr, fCurr, deltaCurr):
        self.__xPath.append(xCurr)
        self.__fPath.append(fCurr)

        self.__deltaList.append(deltaCurr)

    def __get_init_point(self):
        if self.__init_point == None:
            return self.__init_seed(self.__seed)
        else:
            return np.array(self.__init_point)


import unittest

class TestClass(unittest.TestCase):
    def setUp(self):
        print()
        print('*'*5 + 'Test Begins' + '*'*5)

    def test_case1(self):
        obj = BFGS_With_Trust_Region(init_point=[[0.3], [0.4]], func_idx=1)
        obj.solve()
        self.assertEqual(obj.is_converged(), True)

    def test_case2(self):
        obj = BFGS_With_Trust_Region(func_idx=2)
        obj.solve()
        self.assertEqual(obj.is_converged(), True)

    def tearDown(self):
        print('*'*5 + 'Test Over' + '*'*5)
        print()


if __name__ == '__main__':
    unittest.main()
