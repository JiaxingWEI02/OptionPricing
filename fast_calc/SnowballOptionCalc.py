import numpy as np
import cupy as cp
from scipy.sparse.linalg import splu
from scipy import sparse
from cupyx.scipy import sparse as cp_sparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


#####################
# v1.0: 提供蒙特卡洛模拟和有限差分法对雪球期权进行定价, 
# 不具备实际使用价值, 仅用作对比gpu和cpu在处理大量for循环时的速度
# 使用的定价公式均为Black-Scholes公式
#####################

class SnowBallOption:
    def __init__(self, params, method='monte_carlo', device='cpu'):
        self.S0 = params["S0"]
        self.K = params["K"]
        self.r = params["r"]
        self.sigma = params["sigma"]
        self.T = params["T"]
        self.steps = params["steps"]
        self.trials = params["trials"]
        self.method = method
        self.grid_points = params["grid_points"]
        self.omega = 1                             # 全当作call option
        
        if device == 'gpu':
            self.xp = cp
        elif device == 'cpu':
            self.xp = np
        else:
            raise TypeError("Avaiable device is 'gpu' or 'cpu'")

    def get_method(self):
        return self.method

    def monte_carlo(self):
        dt = self.T / self.steps
        paths = self.xp.zeros((self.steps, self.trials))
        paths[0] = self.S0
        for t in range(1, self.steps):
            z = self.xp.random.standard_normal(self.trials)
            paths[t] = paths[t-1] * self.xp.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * self.xp.sqrt(dt) * z)

        if isinstance(paths, cp.ndarray):
            payoff = cp.maximum(paths[-1] - self.K, 0)
            return cp.exp(-self.r * self.T) * cp.mean(payoff)
        elif isinstance(paths, np.ndarray):
            payoff = np.maximum(paths[-1] - self.K, 0)
            return np.exp(-self.r * self.T) * np.mean(payoff)
        else:
            raise TypeError('Wrong Input Type')
        
    
    # implict fd
    def fdm(self):
        S_max = 3 * self.xp.maximum(self.S0, self.K)
        S_min = 1/3 * self.xp.minimum(self.S0, self.K)

        x0 = self.xp.log(self.S0)
        x_max = self.xp.log(S_max)  
        x_min = self.xp.log(S_min)

        x_vec, dx = self.xp.linspace(x_min, x_max, num = self.steps, retstep = True)       #空间离散化
        t_vec, dt = self.xp.linspace(0, self.T, num = self.trials, retstep = True)         #时间离散化
        
        self.V = self.xp.zeros((self.steps, self.trials)) #网格构建 
        offset = self.xp.zeros(self.steps-2)              #边界条件

        self.V[0, :] = self.xp.maximum(self.omega * (S_min - self.K * self.xp.exp(-self.r * t_vec)), 0)  #边界条件：S=Smin时，必行权
        self.V[-1, :] = self.xp.maximum(self.omega * (S_max - self.K * self.xp.exp(-self.r * t_vec)), 0) #边界条件：S=Smax时，必不行权          
        self.V[:,-1] = self.xp.maximum(self.omega * (self.xp.exp(x_vec) - self.K), 0)                    #终值条件,t = T

        sigma2 = self.sigma ** 2
        dxx = dx **2

        #coefficient of matrix
        a = ( (dt/2) * ( (self.r-0.5*sigma2)/dx - sigma2/dxx ) )
        b = ( 1 + dt * ( sigma2/dxx + self.r ) )
        c = (-(dt/2) * ( (self.r-0.5*sigma2)/dx + sigma2/dxx ) )

        # distinguish the array type
        # tri-diagnal matrix
        if isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray) and isinstance(b, cp.ndarray):
            self.D = cp_sparse.diags([a, b, c], [-1, 0, 1], shape=(self.steps-2, self.steps-2)).tocsc()
        else:
            self.D = sparse.diags([a, b, c], [-1, 0, 1], shape=(self.steps-2, self.steps-2)).tocsc()

        # Backward iteration
        # 可以考虑用Thomas算法与spsolve,经比较splu应该是最快的
        DD = splu(self.D)
        for i in range(self.trials-2,-1,-1):
            offset[0] = a * self.V[0,i]
            offset[-1] = c * self.V[-1,i]; 
            self.V[1:-1,i] = DD.solve(self.V[1:-1,i+1] - offset) 
            
        return self.xp.interp(x0, x_vec, self.V[:,0])  #线性插值取出所求期权价格
    
    def calc_opt_prc(self):
        if self.method == 'monte_carlo':
            return self.monte_carlo()
        elif self.method == 'fdm':
            return self.fdm()
        else:
            raise ValueError('This version supports "monte_carlo", "fdm" methods')