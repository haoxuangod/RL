#基础DQN网络结构
import math
import operator
import time
from functools import reduce
import torch.nn.functional as F
import torch
from torch import nn
from torch.distributions import constraints, Categorical, MultivariateNormal

from RL.TorchModels.BaseNet import BaseNet
from RL.TorchModels.ModelInput import MLPInput


class DQNOutput(BaseNet):
    """
    input_shape: int 或 长度为1的 list/tuple, 表示特征维度
    output_shape:
      - int: 输出 (B, n)
      - 1-tuple (n,): 输出 (B, 1, n)
      - 2-tuple (H, W): 输出 (B, H, W)
    """
    def __init__(self, input_shape, output_shape,name=None):
        super(DQNOutput, self).__init__(name=name)
        # —— 1. 规范化 input_shape —— #
        if isinstance(input_shape, int):
            in_dim = input_shape
        elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            in_dim = input_shape[0]
        else:
            raise ValueError(
                f"input_shape must be int or length-1 list/tuple, got {input_shape}"
            )
        self.in_dim = in_dim
        self.input_shape = input_shape
        # —— 2. 规范化 output_shape —— #
        if isinstance(output_shape, int):
            # 单一路输出
            self.output_shape = (output_shape,)
        elif isinstance(output_shape, (list, tuple)) and 1 <= len(output_shape) <= 2:
            if len(output_shape) == 1:
                # (n,) -> (1, n)
                self.output_shape = (1, output_shape[0])
            else:
                # 直接 (H, W)
                self.output_shape = tuple(output_shape)
        else:
            raise ValueError(
                f"output_shape must be int or length-1/2 tuple, got {output_shape}"
            )

        # —— 3. 构建 head —— #
        if isinstance(output_shape, int):
            # 情况1：用户传 int，或者视作单一路输出
            self.head = nn.Linear(in_dim, output_shape)
            self._multi = False
        else:
            H, W = self.output_shape  # 解包：(H, W)
            # 多路输出：H 条平行的 Linear，每条输出 W 维
            self.heads = nn.ModuleList([nn.Linear(in_dim, W) for _ in range(H)])
            self._multi = True

    def forward(self, x):
        """
        x: (B, in_dim)
        返回:
          - 单一路: (B, W)
          - 多路:     (B, H, W)
        """
        flag=False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            flag=True
        elif len(x.shape) == 2:
            pass
        else:
            raise ValueError("x.shape must be length-1/2 tuple [in_dim,]、[batch_size,in_dim]")
        if not self._multi:
            # 单一路直接返回 (B, W)
            return self.head(x)
        # 多路：对每个子 head 都跑一遍
        outs = [head(x) for head in self.heads]  # list of (B, W), len=H
        # 堆叠到 (B, W, H)，再 permute -> (B, H, W)
        out = torch.stack(outs, dim=-1).permute(0, 2, 1)
        if flag:
            out = out.squeeze(0)
        return out




def is_lower_triangular(tensor: torch.Tensor,
                        rtol: float = 1e-5,
                        atol: float = 1e-8) -> bool:
    """
    判断 tensor 是否为下三角矩阵（包含主对角线）。

    对于浮点张量，使用 allclose，可以容忍微小数值误差；
    对于整数或精确需求，可改为使用 torch.equal。
    """
    tril = torch.tril(tensor)
    if tensor.dtype.is_floating_point:
        return torch.allclose(tensor, tril, rtol=rtol, atol=atol)
    else:
        return torch.equal(tensor, tril)

class PopArt(BaseNet):
    """
    PopArt layer for adaptive normalization of value targets.
    Maintains running mean and std of targets and rescales output layer.
    """
    def __init__(self, input_dim,name=None,beta=3e-4, eps=1e-5):
        super().__init__(name=name)
        self.beta = beta
        self.eps = eps
        # running statistics for the output values
        #32为batch_size，需要修改
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        # linear layer mapping features to normalized prediction u(s)
        self.linear = nn.Linear(input_dim, 1)
        self.first_update_flag=False
        #nn.init.orthogonal_(self.linear.weight, gain=1e-2)
        #nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        # x: feature tensor [B, input_dim]
        u = self.linear(x)
        if not (u.abs()<100).all():
            pass
        '''
        assert (u.abs()<100).all()
        assert (self.sigma.abs()<5).all()
        assert (self.mu.abs()<5).all()
        assert ((u*self.sigma).abs()<100).all()
        '''
        V = u.squeeze(-1) * self.sigma + self.mu # scale to true value range
        return V.unsqueeze(-1), u

    def update_stats(self, targets):
        # targets: tensor [B, 1]
        batch_mu = targets.mean(dim=0)
        batch_var = targets.var(dim=0, unbiased=False)
        # old stats
        mu_old = self.mu.clone()
        sigma_old = self.sigma.clone()
        if self.first_update_flag:
            new_mu = batch_mu
            new_var= batch_var
            new_sigma = torch.sqrt(new_var + self.eps)
        else:
            # new stats
            new_mu = (1 - self.beta) * mu_old + self.beta * batch_mu
            new_var = (1 - self.beta) * (sigma_old ** 2) + self.beta * batch_var
            new_sigma = torch.sqrt(new_var + self.eps)

        # adjust linear weights & bias to preserve output
        w = self.linear.weight.data       # [1, input_dim]
        b = self.linear.bias.data         # [1]
        # scale weight rows and adjust bias
        self.linear.weight.data = w * (sigma_old / new_sigma).unsqueeze(1)
        self.linear.bias.data = (b * sigma_old + mu_old - new_mu) / new_sigma
        # assign new stats
        self.mu.data = new_mu
        self.sigma.data = new_sigma

class NAFPopArtOutput(BaseNet):
    """
       Normalized Advantage Function Network supporting vector actions via ModuleList.
    """

    def __init__(self, input_shape, output_shape,name=None,a_max=1,a_min=-1,beta=3e-4):
        '''

        :param input_shape:
        :param output_shape: 必须为一个int或者一维的tuple/list,如果是int则输出[action_dim,]否则输出[action_dim,1]
        '''
        super().__init__(name=name)
        if isinstance(input_shape, int):
            in_dim = input_shape
        elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            in_dim = input_shape[0]
        else:
            raise ValueError(
                f"input_shape must be int or length-1 list/tuple, got {input_shape}"
            )
        self.in_dim = in_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.a_max = a_max
        self.a_min=a_min
        if isinstance(output_shape, int):
            self.output_dim = output_shape
        elif isinstance(output_shape,(tuple, list)) and len(output_shape) == 1:
            self.output_dim =output_shape[0]
        else:
            raise ValueError( f"output_shape must be int or length-1 list/tuple, got {output_shape}")
        self.mu_heads = nn.ModuleList([
            nn.Linear(self.in_dim, 1) for _ in range(self.output_dim)
        ])
        # PopArt for value head: input hidden_dim, outputs normalized u
        self.popart = PopArt(input_dim=self.in_dim, beta=beta,tensorboard_log_dir=self.tensorboard_log_dir)
        # Lower-triangular entries for L matrix (for advantage quadratic form)
        self.l_entry = nn.Linear(self.in_dim, self.output_dim * (self.output_dim + 1) // 2)
        '''
        # 权重初始化与论文一致: 小范围正态分布
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        '''
        # Precompute lower-triangular indices (row, col) and diagonal mask for entries
        rows, cols = torch.tril_indices(self.output_dim, self.output_dim)
        self.register_buffer('tril_rows', rows)
        self.register_buffer('tril_cols', cols)
        # Boolean mask indicating which entries are diagonal (row == col)
        diag_mask = (rows == cols)
        self.register_buffer('diag_mask', diag_mask)
        self.cnt=0
        self.q_cnt=0
    def build_L(self,entries, batch_size, dim, device, clip_value=10.0):
        """
        entries: Tensor of shape [batch, dim*(dim+1)//2]
        返回: 稳定的 L 矩阵 [batch, dim, dim]
        """
        L = torch.zeros(batch_size, dim, dim, device=device)
        jitter=2e-6
        entries=torch.tanh(entries)

        # Process entries: softplus on diagonal, clamp off-diagonals
        entries_processed = torch.where(
            self.diag_mask.unsqueeze(0),
            F.softplus(entries, beta=1.0, threshold=20.0)+jitter,
            entries.clamp(-clip_value, clip_value)
        )
        
        L[:, self.tril_rows, self.tril_cols] = entries_processed

        #L[:, self.tril_rows, self.tril_cols] = entries+jitter
        #L.diagonal(dim1=1, dim2=2).exp_()

        # 清理任何残余的 inf/nan
        L = torch.nan_to_num(L, nan=1e-8, posinf=1e6, neginf=-1e6)

        return L
    def forward(self, x: torch.Tensor,actions=None):
        '''
        x: [batch_size, input_dim]有batch_size就会输出bactch_size这一维度:[batch_size,output_dim,?]
            [input_dim] [output_dim,?]
        :param x:
        :return:
        '''

        self.cnt=self.cnt+1
        flag=False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            flag=True

        V,u = self.popart(x)
        V=V.squeeze(-1)
        # Compute mu for each action dimension and stack
        mu_list = [head(x) for head in self.mu_heads]  # list of [batch, 1]
        mu = torch.cat(mu_list, dim=1)  # [batch, action_dim]
        mu = torch.tanh(mu)  # bound actions to [-1,1]
        a_min = torch.tensor(self.a_min, device=x.device)
        a_max = torch.tensor(self.a_max, device=x.device)
        mu=a_min + (mu + 1) * (a_max - a_min) / 2 # bound actions to [a_min,a_max]
        # Build lower-triangular L for advantage
        batch_size = x.size(0)
        dim = self.output_dim
        device = next(self.parameters()).device
        L=self.build_L(self.l_entry(x), batch_size, self.output_dim,device, clip_value=10.0)
        # 提取各批次矩阵的对角线
        # 如果 L 形状是 (batch, dim, dim)，diag.shape 就是 (batch, dim)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)

        # 找出最小值和是否存在 ≤ 0 的元素
        min_vals, _ = diag.min(dim=1)
        #print("每个样本的最小对角元素：", min_vals)
        assert  not (min_vals <= 0).any().item()
        #assert is_lower_triangular(L)
        # Precision matrix

        P = L@L.transpose(-1, -2)  # [batch, dim, dim]

        P=0.5 * (P + P.transpose(-1, -2))

        I = torch.eye(dim, device=P.device, dtype=P.dtype).unsqueeze(0)
        self.matrix_eps=2e-6
        P = P + self.matrix_eps * I

        try:
            _ = torch.linalg.cholesky(P)  # PyTorch 1.8 以上推荐用 torch.linalg.cholesky

        except RuntimeError as e:
            with torch.no_grad():

                eigvals = torch.linalg.eigvalsh(P)  # [batch, dim] 的实对称矩阵特征值
                min_eig = eigvals.min(dim=1).values  # 每个样本的最小特征值
                print("P 最小特征值 =", min_eig)
            print("P 不是正定的：", e)
        '''
        Sigma_raw = torch.inverse(P1)
        # 强制对称：
        Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))
        try:
            _ = torch.linalg.cholesky(Sigma)  # PyTorch 1.8 以上推荐用 torch.linalg.cholesky

        except RuntimeError as e:
            with torch.no_grad():
                eigvals = torch.linalg.eigvalsh(Sigma)  # [batch, dim] 的实对称矩阵特征值
                min_eig = eigvals.min(dim=1).values  # 每个样本的最小特征值
                print("P^-1 最小特征值 =", min_eig)
            print("P^-1 不是正定的：", e)
        '''
        if isinstance(self.output_shape,(tuple,list)):
            mu=mu.unsqueeze(-1)
        if flag==True:
            mu=mu.squeeze(0)
        Q=None
        if actions is not None:
            diff = (actions - mu)
            # A: [B,1,D], B: [B,D,D]
            # 我们对 D 这个维度求和，留下 batch,1,K,D
            tmp = torch.einsum('bld,bdm->blm', diff.unsqueeze(1), P)  # [32,1,29]
            # A:[B,l,M] B:[B,M,1]
            tmp1 = torch.einsum('blm,bma->bla', tmp, diff.unsqueeze(2))
            A = -0.5 * tmp1  # [1,1,1]
            A = A.squeeze(-1).squeeze(-1)


            Q = V + A
            self.q_cnt=self.q_cnt + 1

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(f'V/mean',V.mean(),self.cnt)

            if Q is not None:
                self.summary_writer.add_scalar(f'Q/mean',Q.mean(),self.q_cnt)

        return mu,V,Q,P


class NAFOutput(BaseNet):
    """
       Normalized Advantage Function Network supporting vector actions via ModuleList.
    """

    def __init__(self, input_shape, output_shape,  name=None, a_max=1, a_min=-1, beta=3e-4):
        '''

        :param input_shape:
        :param output_shape: 必须为一个int或者一维的tuple/list,如果是int则输出[action_dim,]否则输出[action_dim,1]
        '''
        super().__init__(name=name)
        if isinstance(input_shape, int):
            in_dim = input_shape
        elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            in_dim = input_shape[0]
        else:
            raise ValueError(
                f"input_shape must be int or length-1 list/tuple, got {input_shape}"
            )
        self.in_dim = in_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.a_max = a_max
        self.a_min = a_min
        if isinstance(output_shape, int):
            self.output_dim = output_shape
        elif isinstance(output_shape, (tuple, list)) and len(output_shape) == 1:
            self.output_dim = output_shape[0]
        else:
            raise ValueError(f"output_shape must be int or length-1 list/tuple, got {output_shape}")
        self.mu_heads = nn.ModuleList([
            nn.Linear(self.in_dim, 1) for _ in range(self.output_dim)
        ])
        # State value branch
        self.v = nn.Linear(self.in_dim, 1)
        # Lower-triangular entries for L matrix (for advantage quadratic form)
        self.l_entry = nn.Linear(self.in_dim, self.output_dim * (self.output_dim + 1) // 2)
        # —— 关键：把 weight 全部置零 ——
        #nn.init.uniform_(self.l_entry.weight, -1e-2, 1e-2)
        # 先把所有 bias 置零
        #nn.init.zeros_(self.l_entry.bias)
        # Precompute lower-triangular indices (row, col) and diagonal mask for entries
        rows, cols = torch.tril_indices(self.output_dim, self.output_dim)
        self.register_buffer('tril_rows', rows)
        self.register_buffer('tril_cols', cols)
        # Boolean mask indicating which entries are diagonal (row == col)
        diag_mask = (rows == cols)
        diag_pos = diag_mask.nonzero(as_tuple=True)[0]
        '''
        init_std=0.1
        # 期望 softplus(b) + eps = target_Ldiag
        target_Ldiag = math.sqrt(1.0 / (init_std ** 2))  # = 1/σ = 10
        # softplus^{-1}(x) = log(exp(x) - 1)
        b_init = math.log(math.exp(target_Ldiag) - 1.0)
        
        
        b_init=0.2
        
        # 把这些偏置项设为 b_init
        with torch.no_grad():
            self.l_entry.bias[diag_pos] = b_init

        
         # 4) **关键**：为 weight 设一个与 b_init 同量级的 std
        std = target_Ldiag / math.sqrt(in_dim)
        nn.init.normal_(self.l_entry.weight, mean=0.0, std=0.2)
        print(b_init)
        '''
        self.register_buffer('diag_mask', diag_mask)
        self.cnt = 0
        self.q_cnt = 0
    def build_L(self, entries, batch_size, dim, device, clip_value=10.0):
        """
        entries: Tensor of shape [batch, dim*(dim+1)//2]
        返回: 稳定的 L 矩阵 [batch, dim, dim]
        """
        L = torch.zeros(batch_size, dim, dim, device=device)
        jitter = 2e-6
        entries = torch.tanh(entries)

        # Process entries: softplus on diagonal, clamp off-diagonals
        entries_processed = torch.where(
            self.diag_mask.unsqueeze(0),
            F.softplus(entries, beta=1.0, threshold=20.0) + jitter,
            entries.clamp(-clip_value, clip_value)
        )

        L[:, self.tril_rows, self.tril_cols] = entries_processed

        # L[:, self.tril_rows, self.tril_cols] = entries+jitter
        # L.diagonal(dim1=1, dim2=2).exp_()

        # 清理任何残余的 inf/nan
        L = torch.nan_to_num(L, nan=1e-8, posinf=1e6, neginf=-1e6)

        return L

    def forward(self, x: torch.Tensor, actions=None):
        '''
        x: [batch_size, input_dim]有batch_size就会输出bactch_size这一维度:[batch_size,output_dim,?]
            [input_dim] [output_dim,?]
        :param x:
        :return:
        '''
        self.cnt = self.cnt + 1
        flag = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            flag = True

        V=self.v(x).squeeze(-1)
        # Compute mu for each action dimension and stack
        mu_list = [head(x) for head in self.mu_heads]  # list of [batch, 1]
        mu = torch.cat(mu_list, dim=1)  # [batch, action_dim]
        tanh_mu = torch.tanh(mu)  # bound actions to [-1,1]
        a_min = torch.tensor(self.a_min, device=x.device)
        a_max = torch.tensor(self.a_max, device=x.device)
        tanh_mu = a_min + (tanh_mu + 1) * (a_max - a_min) / 2  # bound actions to [a_min,a_max]
        # Build lower-triangular L for advantage
        batch_size = x.size(0)
        dim = self.output_dim
        device = next(self.parameters()).device
        L = self.build_L(self.l_entry(x), batch_size, self.output_dim, device, clip_value=10.0)
        # 提取各批次矩阵的对角线
        # 如果 L 形状是 (batch, dim, dim)，diag.shape 就是 (batch, dim)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)

        # 找出最小值和是否存在 ≤ 0 的元素
        min_vals, _ = diag.min(dim=1)
        # print("每个样本的最小对角元素：", min_vals)
        assert not (min_vals <= 0).any().item()
        # assert is_lower_triangular(L)
        # Precision matrix

        P = L @ L.transpose(-1, -2)  # [batch, dim, dim]

        P = 0.5 * (P + P.transpose(-1, -2))
        I = torch.eye(dim, device=P.device, dtype=P.dtype).unsqueeze(0)
        self.matrix_eps = 2e-6
        P = P + self.matrix_eps * I


        try:
            _ = torch.linalg.cholesky(P)  # PyTorch 1.8 以上推荐用 torch.linalg.cholesky

        except RuntimeError as e:
            with torch.no_grad():

                eigvals = torch.linalg.eigvalsh(P)  # [batch, dim] 的实对称矩阵特征值
                min_eig = eigvals.min(dim=1).values  # 每个样本的最小特征值
                print("P 最小特征值 =", min_eig)
            print("P 不是正定的：", e)
        '''
        Sigma_raw = torch.inverse(P1)
        # 强制对称：
        Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))
        try:
            _ = torch.linalg.cholesky(Sigma)  # PyTorch 1.8 以上推荐用 torch.linalg.cholesky

        except RuntimeError as e:
            with torch.no_grad():
                eigvals = torch.linalg.eigvalsh(Sigma)  # [batch, dim] 的实对称矩阵特征值
                min_eig = eigvals.min(dim=1).values  # 每个样本的最小特征值
                print("P^-1 最小特征值 =", min_eig)
            print("P^-1 不是正定的：", e)
        '''
        if isinstance(self.output_shape, (tuple, list)):
            mu = mu.unsqueeze(-1)
        if flag == True:
            mu = mu.squeeze(0)
        Q = None
        if actions is not None:
            diff = (actions - tanh_mu)
            # A: [B,1,D], B: [B,D,D]
            # 我们对 D 这个维度求和，留下 batch,1,K,D
            tmp = torch.einsum('bld,bdm->blm', diff.unsqueeze(1), P)  # [32,1,29]
            # A:[B,l,M] B:[B,M,1]
            tmp1 = torch.einsum('blm,bma->bla', tmp, diff.unsqueeze(2))
            A = -0.5 * tmp1  # [1,1,1]
            A = A.squeeze(-1).squeeze(-1)

            Q = V + A
            self.q_cnt = self.q_cnt + 1

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(f'{self.base_tag}/V/mean', V.mean(), self.cnt)
            #self.summary_writer.add_scalar(f'alpha/value', self.alpha, self.cnt)
            if Q is not None:
                self.summary_writer.add_scalar(f'{self.base_tag}/Q/mean', Q.mean(), self.q_cnt)

        return mu, V, Q, P


class BNAFOutput(BaseNet):
    """
       Normalized Advantage Function Network supporting vector actions via ModuleList.
    """

    def __init__(self, input_shape, output_shape, name=None, a_max=1, a_min=-1, beta=3e-4):
        '''

        :param input_shape:
        :param output_shape: 必须为一个int或者一维的tuple/list,如果是int则输出[action_dim,]否则输出[action_dim,1]
        '''
        super().__init__( name=name)
        if isinstance(input_shape, int):
            in_dim = input_shape
        elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            in_dim = input_shape[0]
        else:
            raise ValueError(
                f"input_shape must be int or length-1 list/tuple, got {input_shape}"
            )
        self.in_dim = in_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.a_max = a_max
        self.a_min = a_min
        if isinstance(output_shape, int):
            self.output_dim = output_shape
        elif isinstance(output_shape, (tuple, list)) and len(output_shape) == 1:
            self.output_dim = output_shape[0]
        else:
            raise ValueError(f"output_shape must be int or length-1 list/tuple, got {output_shape}")
        self.mu_heads = nn.ModuleList([
            nn.Linear(self.in_dim, 1) for _ in range(self.output_dim)
        ])
        # State value branch
        self.v = nn.Linear(self.in_dim, 1)
        # Lower-triangular entries for L matrix (for advantage quadratic form)
        self.l_entry = nn.Linear(self.in_dim, self.output_dim * (self.output_dim + 1) // 2)
        # —— 关键：把 weight 全部置零 ——
        # nn.init.uniform_(self.l_entry.weight, -1e-2, 1e-2)
        # 先把所有 bias 置零
        nn.init.zeros_(self.l_entry.bias)
        # Precompute lower-triangular indices (row, col) and diagonal mask for entries
        rows, cols = torch.tril_indices(self.output_dim, self.output_dim)
        self.register_buffer('tril_rows', rows)
        self.register_buffer('tril_cols', cols)
        # Boolean mask indicating which entries are diagonal (row == col)
        diag_mask = (rows == cols)
        diag_pos = diag_mask.nonzero(as_tuple=True)[0]
        '''
        init_std = 0.1
        # 期望 softplus(b) + eps = target_Ldiag
        target_Ldiag = math.sqrt(1.0 / (init_std ** 2))  # = 1/σ = 10
        # softplus^{-1}(x) = log(exp(x) - 1)
        b_init = math.log(math.exp(target_Ldiag) - 1.0)
        b_init = 0.2

        # 把这些偏置项设为 b_init
        with torch.no_grad():
            self.l_entry.bias[diag_pos] = b_init

        
         # 4) **关键**：为 weight 设一个与 b_init 同量级的 std
        std = target_Ldiag / math.sqrt(in_dim)
        
        nn.init.normal_(self.l_entry.weight, mean=0.0, std=0.2)
        print(b_init)
        '''
        self.register_buffer('diag_mask', diag_mask)
        self.cnt = 0
        self.q_cnt = 0

    def build_L(self, entries, batch_size, dim, device, clip_value=10.0):
        """
        entries: Tensor of shape [batch, dim*(dim+1)//2]
        返回: 稳定的 L 矩阵 [batch, dim, dim]
        """
        L = torch.zeros(batch_size, dim, dim, device=device)
        jitter = 2e-6
        entries = torch.tanh(entries)

        # Process entries: softplus on diagonal, clamp off-diagonals
        entries_processed = torch.where(
            self.diag_mask.unsqueeze(0),
            F.softplus(entries, beta=1.0, threshold=20.0) + jitter,
            entries.clamp(-clip_value, clip_value)
        )

        L[:, self.tril_rows, self.tril_cols] = entries_processed

        # L[:, self.tril_rows, self.tril_cols] = entries+jitter
        # L.diagonal(dim1=1, dim2=2).exp_()

        # 清理任何残余的 inf/nan
        L = torch.nan_to_num(L, nan=1e-8, posinf=1e6, neginf=-1e6)

        return L

    def forward(self, x: torch.Tensor, actions=None):
        '''
        x: [batch_size, input_dim]有batch_size就会输出bactch_size这一维度:[batch_size,output_dim,?]
            [input_dim] [output_dim,?]
        :param x:
        :return:
        '''
        self.cnt = self.cnt + 1
        flag = False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            flag = True

        V = self.v(x).squeeze(-1)
        # Compute mu for each action dimension and stack
        mu_list = [head(x) for head in self.mu_heads]  # list of [batch, 1]
        mu = torch.cat(mu_list, dim=1)  # [batch, action_dim]

        # Build lower-triangular L for advantage
        batch_size = x.size(0)
        dim = self.output_dim
        device = next(self.parameters()).device
        L = self.build_L(self.l_entry(x), batch_size, self.output_dim, device, clip_value=10.0)
        # 提取各批次矩阵的对角线
        # 如果 L 形状是 (batch, dim, dim)，diag.shape 就是 (batch, dim)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)

        # 找出最小值和是否存在 ≤ 0 的元素
        min_vals, _ = diag.min(dim=1)
        # print("每个样本的最小对角元素：", min_vals)
        assert not (min_vals <= 0).any().item()
        # assert is_lower_triangular(L)
        # Precision matrix

        P = L @ L.transpose(-1, -2)  # [batch, dim, dim]

        P = 0.5 * (P + P.transpose(-1, -2))
        I = torch.eye(dim, device=P.device, dtype=P.dtype).unsqueeze(0)
        self.matrix_eps = 2e-6
        P = P + self.matrix_eps * I

        try:
            _ = torch.linalg.cholesky(P)  # PyTorch 1.8 以上推荐用 torch.linalg.cholesky

        except RuntimeError as e:
            with torch.no_grad():

                eigvals = torch.linalg.eigvalsh(P)  # [batch, dim] 的实对称矩阵特征值
                min_eig = eigvals.min(dim=1).values  # 每个样本的最小特征值
                print("P 最小特征值 =", min_eig)
            print("P 不是正定的：", e)
        '''
        Sigma_raw = torch.inverse(P1)
        # 强制对称：
        Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))
        try:
            _ = torch.linalg.cholesky(Sigma)  # PyTorch 1.8 以上推荐用 torch.linalg.cholesky

        except RuntimeError as e:
            with torch.no_grad():
                eigvals = torch.linalg.eigvalsh(Sigma)  # [batch, dim] 的实对称矩阵特征值
                min_eig = eigvals.min(dim=1).values  # 每个样本的最小特征值
                print("P^-1 最小特征值 =", min_eig)
            print("P^-1 不是正定的：", e)
        '''
        if isinstance(self.output_shape, (tuple, list)):
            mu = mu.unsqueeze(-1)
        if flag == True:
            mu = mu.squeeze(0)
        a_min = torch.tensor(self.a_min, device=x.device)
        a_max = torch.tensor(self.a_max, device=x.device)
        #直接截断会无法训练
        clip_mu=torch.clip(mu,-1,1)
        clip_mu = a_min + (clip_mu + 1) * (a_max - a_min) / 2
        Q = None
        if actions is not None:
            diff = (actions - mu)
            diff_clip = clip_mu - mu
            first=-torch.bmm(diff.unsqueeze(1), torch.bmm(P, diff.unsqueeze(2))).squeeze(-1)
            second=torch.bmm(diff_clip.unsqueeze(1), torch.bmm(P, diff_clip.unsqueeze(2))).squeeze(-1)
            A=0.5*(first+second)
            Q = V + A
            self.q_cnt = self.q_cnt + 1

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(f'V/mean', V.mean(), self.cnt)
            # self.summary_writer.add_scalar(f'alpha/value', self.alpha, self.cnt)
            if Q is not None:
                self.summary_writer.add_scalar(f'Q/mean', Q.mean(), self.q_cnt)

        return clip_mu, V, Q, P
class MAFOutput(BaseNet):
    def __init__(self, input_shape, output_shape, name=None, a_max=1, a_min=-1,num_components=4):
        """
        state_dim: 状态维度
        output_dim: 动作维度
        num_components: 高斯混合分量数 K
        """
        super().__init__(name=name)
        super().__init__(name=name)
        if isinstance(input_shape, int):
            in_dim = input_shape
        elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            in_dim = input_shape[0]
        else:
            raise ValueError(
                f"input_shape must be int or length-1 list/tuple, got {input_shape}"
            )
        self.in_dim = in_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.a_max = a_max
        self.a_min = a_min
        if isinstance(output_shape, int):
            self.output_dim = output_shape
        elif isinstance(output_shape, (tuple, list)) and len(output_shape) == 1:
            self.output_dim = output_shape[0]
        else:
            raise ValueError(f"output_shape must be int or length-1 list/tuple, got {output_shape}")
        self.a_max=a_max
        self.a_min=a_min
        self.K = num_components

        '''
        # 公共特征提取网络
        self.feat = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        '''
        # V(s) 头
        self.v_head = nn.Linear(in_dim, 1)
        # 混合权重 logits -> softmax 得到 w_k(s)
        self.logits_head = nn.Linear(in_dim, num_components)
        # 每个分量的 μ_k(s)
        self.mu_heads =nn.Linear(self.in_dim, self.output_dim*self.K)

        # 每个分量的 L_k(s) 下三角展平
        self.l_heads = nn.Linear(self.in_dim, self.K*self.output_dim*(self.output_dim+1)//2)
        # Precompute lower-triangular indices (row, col) and diagonal mask for entries
        rows, cols = torch.tril_indices(self.output_dim, self.output_dim)
        self.register_buffer('tril_rows', rows)
        self.register_buffer('tril_cols', cols)
        # Boolean mask indicating which entries are diagonal (row == col)
        diag_mask = (rows == cols)
        self.register_buffer('diag_mask', diag_mask)
    def build_L(self, entries, batch_size, dim,K,device, clip_value=10.0):
        """
        entries: Tensor of shape [batch, dim*(dim+1)//2]
        返回: 稳定的 L 矩阵 [batch, dim, dim]
        """
        L = torch.zeros(batch_size,K,dim, dim, device=device)
        jitter = 2e-6
        entries = torch.tanh(entries)
        diag_mask=self.diag_mask.expand_as(entries)
        # Process entries: softplus on diagonal, clamp off-diagonals
        entries_processed = torch.where(
            diag_mask,
            F.softplus(entries, beta=1.0, threshold=20.0) + jitter,
            entries.clamp(-clip_value, clip_value)
        )

        L[:,:, self.tril_rows, self.tril_cols] = entries_processed

        # L[:, self.tril_rows, self.tril_cols] = entries+jitter
        # L.diagonal(dim1=1, dim2=2).exp_()

        # 清理任何残余的 inf/nan
        L = torch.nan_to_num(L, nan=1e-8, posinf=1e6, neginf=-1e6)

        return L
    def forward(self, x, actions=None,n_samples=10):
        """
        计算 Q(s,a) = V(s) + A(s,a), 其中
        A(s,a) = sum_k w_k * A_k(s,a),
        A_k(s,a) = -1/2 (a - μ_k)^T P_k (a - μ_k), P_k = L_k L_k^T.
        应该拆开
        """
        batch_size = x.shape[0]
        device = next(self.parameters()).device
        V = self.v_head(x).view(batch_size)  # [B]
        logits = self.logits_head(x)  # [B,K]
        w = F.softmax(logits, dim=-1)  # [B,K]
        Qs = []
        mus=[]
        Ps=[]

        mus=self.mu_heads(x).view(-1, self.K, self.output_dim)
        l_entries=self.l_heads(x).view(-1, self.K, self.output_dim*(self.output_dim+1)//2)
        #1ms
        L=self.build_L(l_entries,batch_size, self.output_dim,self.K,device, clip_value=10.0)
        Ps = L @ L.transpose(-1, -2)
        Ps = 0.5 * (Ps + Ps.transpose(-1, -2))
        I = torch.eye(self.output_dim, device=Ps.device, dtype=Ps.dtype).unsqueeze(0).unsqueeze(0)
        self.matrix_eps = 2e-6
        Ps = Ps + self.matrix_eps * I
        '''
        for k in range(self.K):
            mu = self.mu_heads[k](x)          # [B, D]
            # 构造 L_k
            l_entries = self.l_heads[k](x)       # [B, D*(D+1)/2]
            L = self.build_L(l_entries,batch_size, self.output_dim, device, clip_value=10.0)
            P = L @ L.transpose(-1,-2)          # [B, D, D]
            mus.append(mu)
            Ps.append(P)
        '''
        Q=None
        if actions is not None:
            actions_tmp=actions.unsqueeze(1) #[B,D]->[B,1,D]
            actions_tmp=actions_tmp.expand(-1,self.K,-1)#[B,1,D]->[B,K,D]
            #mus_tmp=torch.stack(mus,dim=1)#[B,K,D]
            mus_tmp=mus
            diff_tmp=(actions_tmp - mus_tmp).unsqueeze(-1)
            #P_tmp=torch.stack(Ps,dim=1)#[B,K,D,D]
            P_tmp=Ps
            '''
            for i in range(self.K):
                P=Ps[i]
                mu=mus[i]
                diff = (actions - mu).unsqueeze(-1)  # [B, D, 1]
                A_k = -0.5 * (diff.transpose(-1,-2) @ (P @ diff)).squeeze(-1).squeeze(-1)  # [B]
                Qs.append(A_k)
            '''
            A=-0.5 * (diff_tmp.transpose(-1,-2) @ (P_tmp @ diff_tmp)).squeeze(-1).squeeze(-1)  # [B,K]
            # combine
            log_pi = torch.logsumexp(torch.log(w + 1e-8) + A, dim=1)  # [B]
            A = log_pi
            #A1 = torch.stack(Qs, dim=1)           # [B, K]
            #A_weighted = (w * A).sum(dim=1)      # [B]

            Q=V+A# [B]

        return mus,Ps,w,V,Q

class NAFOutput1(nn.Module):
    """
       Normalized Advantage Function Network supporting vector actions via ModuleList.
    """

    def __init__(self, input_shape, output_shape,a_max=1,a_min=-1,beta=3e-4):
        '''

        :param input_shape:
        :param output_shape: 必须为一个int或者一维的tuple/list,如果是int则输出[action_dim,]否则输出[action_dim,1]
        '''
        super().__init__()
        if isinstance(input_shape, int):
            in_dim = input_shape
        elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            in_dim = input_shape[0]
        else:
            raise ValueError(
                f"input_shape must be int or length-1 list/tuple, got {input_shape}"
            )
        self.in_dim = in_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.a_max = a_max
        self.a_min=a_min
        if isinstance(output_shape, int):
            self.output_dim = output_shape
        elif isinstance(output_shape,(tuple, list)) and len(output_shape) == 1:
            self.output_dim =output_shape[0]
        else:
            raise ValueError( f"output_shape must be int or length-1 list/tuple, got {output_shape}")
        # State value branch
        self.v = nn.Linear(self.in_dim, 1)
        self.mu_heads = nn.ModuleList([
            nn.Linear(self.in_dim, 1) for _ in range(self.output_dim)
        ])
        # Lower-triangular entries for L matrix (for advantage quadratic form)
        self.l_entry = nn.Linear(self.in_dim, self.output_dim * (self.output_dim + 1) // 2)
        # Precompute lower-triangular indices (row, col) and diagonal mask for entries
        rows, cols = torch.tril_indices(self.output_dim, self.output_dim)
        self.register_buffer('tril_rows', rows)
        self.register_buffer('tril_cols', cols)
        # Boolean mask indicating which entries are diagonal (row == col)
        diag_mask = (rows == cols)
        self.register_buffer('diag_mask', diag_mask)

    def build_L(self,entries, batch_size, dim, device, clip_value=10.0):
        """
        entries: Tensor of shape [batch, dim*(dim+1)//2]
        返回: 稳定的 L 矩阵 [batch, dim, dim]
        """
        L = torch.zeros(batch_size, dim, dim, device=device)
        #entries = torch.tanh(entries)
        jitter=1e-6
        # Process entries: softplus on diagonal, clamp off-diagonals
        entries_processed = torch.where(
            self.diag_mask.unsqueeze(0),
            F.softplus(entries, beta=1.0, threshold=20.0)+jitter,
            entries.clamp(-clip_value, clip_value)
        )

        L[:, self.tril_rows, self.tril_cols] = entries_processed

        # 清理任何残余的 inf/nan
        L = torch.nan_to_num(L, nan=0.0, posinf=1e6, neginf=-1e6)

        return L
    def forward(self, x: torch.Tensor,actions=None):
        '''
        x: [batch_size, input_dim]有batch_size就会输出bactch_size这一维度:[batch_size,output_dim,?]
            [input_dim] [output_dim,?]
        :param x:
        :return:
        '''
        flag=False
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            flag=True

        V = self.v(x)
        # Compute mu for each action dimension and stack
        mu_list = [head(x) for head in self.mu_heads]  # list of [batch, 1]
        mu = torch.cat(mu_list, dim=1)  # [batch, action_dim]
        mu = torch.tanh(mu)  # bound actions to [-1,1]
        a_min = torch.tensor(self.a_min, device=x.device)
        a_max = torch.tensor(self.a_max, device=x.device)
        mu=a_min + (mu + 1) * (a_max - a_min) / 2 # bound actions to [a_min,a_max]
        # Build lower-triangular L for advantage
        batch_size = x.size(0)
        dim = self.output_dim
        device = next(self.parameters()).device
        L=self.build_L(self.l_entry(x), batch_size, self.output_dim,device, clip_value=10.0)
        # 提取各批次矩阵的对角线
        # 如果 L 形状是 (batch, dim, dim)，diag.shape 就是 (batch, dim)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)

        # 找出最小值和是否存在 ≤ 0 的元素
        min_vals, _ = diag.min(dim=1)
        #print("每个样本的最小对角元素：", min_vals)
        assert  not (min_vals <= 0).any().item()
        #assert is_lower_triangular(L)
        # Precision matrix
        P = L @ L.transpose(-1, -2)  # [batch, dim, dim]
        if isinstance(self.output_shape,(tuple,list)):
            mu=mu.unsqueeze(-1)
        if flag==True:
            mu=mu.squeeze(0)
        Q=None
        if actions is not None:
            diff = (actions - mu)
            # A: [B,1,D], B: [B,D,D]
            # 我们对 D 这个维度求和，留下 batch,1,K,D
            tmp = torch.einsum('bld,bdm->blm', diff.unsqueeze(1), P)  # [32,1,29]
            # A:[B,l,M] B:[B,M,1]
            tmp1 = torch.einsum('blm,bma->bla', tmp, diff.unsqueeze(2))
            A = -0.5 * tmp1  # [1,1,1]
            A = A.squeeze(-1)
            Q = V + A

        return mu,V,Q,L

def main():
    data = torch.zeros(291)
    input = MLPInput(input_shape=(291,),output_shape=64)
    output = DQNOutput(input_shape=64,output_shape=(5, 29))
    net = nn.Sequential(input, output)
    print(net(data).shape)
    input = MLPInput(input_shape=(291,), output_shape=64)
    output = NAFOutput(input_shape=64, output_shape=(29,))
    net = nn.Sequential(input, output)
    V,mu,P=net(data)
    print(mu.shape)
if __name__ == "__main__":

    main()


