import torch
import numpy as np
import scipy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from fast_operators import *
from functools import wraps



#####################
# v1.0: 提供蒙特卡洛模拟对期权进行定价, 针对ndarray和torch.tensor类型进行计算
# 期权类型: 欧式期权
# 使用的定价公式均为Black-Scholes公式
#####################


def record_params(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        params = func(*args, **kwargs)
        return params
    return wrapper


class OptionPrc:
    def __init__(self, params, device='cpu'):
        self.type = params["Type"]
        self.S = params["S"]
        self.K = params["K"]
        self.r = params["r"]
        self.sigma = params["sigma"]
        self.T = params["T"]
        self.steps = params["steps"]
        self.trials = params["trials"]
        self.grid_points = params["grid_points"]
        

        if device == 'gpu':
            self._on_cuda_([self.S, self.K, self.r, self.sigma, self.T])


    def _on_cuda_(self, tensor_param_list: list) -> list:
        for x in tensor_param_list:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            else:
                raise TypeError("Only torch.Tensor can be used on GPU")
        
        
    def bs_price(self):
        d_1 = (1 / (self.sigma * sqrt(self.T))) * (log(self.S / self.K) + (self.r + (power(self.sigma, 2) / 2)) * self.T)
        d_2 = d_1 - self.sigma * sqrt(self.T)

        if self.type == "C":
            C = std_norm_cdf(d_1) * self.S - std_norm_cdf(d_2) * self.K * exp(-self.r * self.T)
            return C

        elif self.type == "P":
            P = std_norm_cdf(-d_2) * self.K * exp(-self.r * self.T) - std_norm_cdf(-d_1) * self.S
            return P
        
        else:
            raise WrongInputTypeError
        
    def bs_greeks(self) -> list:
        d_1 = (1 / (self.sigma * sqrt(self.T))) * (log(self.S / self.K) + (self.r + (power(self.sigma, 2) / 2)) * self.T)
        d_2 = d_1 - self.sigma * sqrt(self.T)

        delta = std_norm_cdf(d_1)
        gamma = div(std_norm_pdf(d_1), self.S * self.sigma * sqrt(self.T))
        vega = self.S * std_norm_pdf(d_1) * sqrt(self.T)
        theta = div((self.S * std_norm_pdf(d_1) * self.sigma), (2 * sqrt(self.T))) + self.r * self.K * exp(-self.r * self.T) * std_norm_cdf(d_2)
        rho = self.K * self.T * exp(-self.r * self.T) * std_norm_cdf(d_2)
        
        return [delta, gamma, vega, theta, rho]
    
    def calc_d1(self):
        price = self.bs_price()
        if isinstance(price, torch.Tensor):
            # 计算图不保留, 否则self.S参数在后续高阶导的计算里会不被torch识别
            price.backward(retain_graph=False)
        else:
            raise TypeError("The price should be torch.Tensor type when using autograd method.")
        
        if isinstance(self.S, torch.Tensor):
            delta = self.S.grad
        else:
            raise TypeError("The delta should be torch.Tensor type when using autograd method.")
        
        if isinstance(self.sigma, torch.Tensor):
            vega = self.sigma.grad
        else:
            raise TypeError("The vega should be torch.Tensor type when using autograd method.")
        
        if isinstance(self.T, torch.Tensor):
            theta = self.T.grad
        else:
            raise TypeError("The theta should be torch.Tensor type when using autograd method.")
        
        if isinstance(self.r, torch.Tensor):
            rho = self.r.grad
        else:
            raise TypeError("The rho should be torch.Tensor type when using autograd method.")
        
        return delta, vega, theta, rho
    
    def reset_grad(self, reset_target:Tensor) -> Tensor:
        '''
        在使用autograd计算二阶或更高阶的偏导时, 如果之前的任务计算过参数的梯度, 那么需要将梯度置零
        '''
        for tensor in reset_target:
            tensor.grad = None

    def calc_d2(self):
        if isinstance(self.S, torch.Tensor):
            p = self.bs_price()
            d = torch.autograd.grad(p, self.S, create_graph=True)[0]
            d.backward()
            gamma = self.S.grad
            return gamma
        else:
            raise TypeError("The gamma should be torch.Tensor type when using autograd method.")
        
    def autograd_greeks(self) -> list:
        delta, vega, theta, rho = self.calc_d1()
        # 必须清空一次梯度(重建计算图)才能计算二阶导(原理求解?)
        self.reset_grad([self.S, self.K, self.T, self.sigma, self.r])
        gamma = self.calc_d2()
        # 这一行的目的是重复调用此函数时不会受前面计算的影响
        self.reset_grad([self.S, self.K, self.T, self.sigma, self.r])
        return [delta, gamma, vega, theta, rho]
