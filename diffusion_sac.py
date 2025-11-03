import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Independent, Normal
from typing import Any, Dict, Optional, Tuple, Union

class DiffusionSAC(BasePolicy):
    """
    Implementation of diffusion-based discrete soft actor-critic policy.
    """

    def __init__(
            self,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            dist_fn: Type[torch.distributions.Distribution],
            device: torch.device,
            # alpha: float = 0.05,
            alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
            tau: float = 0.005,
            gamma: float = 0.95,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            pg_coef: float = 1.,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # assert 0.0 <= alpha <= 1.0, "alpha should be in [0, 1]"
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim
            self._action_dim = action_dim

        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(
                self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(
                self._critic_optim, T_max=lr_maxt, eta_min=0.)

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._dist_fn = dist_fn
        # self._alpha = alpha
        self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._pg_coef = pg_coef
        self._device = device

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        obs_next_ = torch.FloatTensor(batch.obs_next).to(self._device)
        next_result = self.forward(
            batch, input="obs_next", model="target_actor")
        dist = next_result.dist
        target_q = self._target_critic.q_min(obs_next_, next_result.act) - dist.log_prob(next_result.act).unsqueeze(1) * self._alpha
        
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm
        )

    def update(
            self,
            sample_size: int,
            buffer: Optional[ReplayBuffer],
            **kwargs: Any
    ) -> Dict[str, Any]:
        if buffer is None: return {}
        self.updating = True
        # sample from replay buffer
        batch, indices = buffer.sample(sample_size)
        # calculate n_step returns
        batch = self.process_fn(batch, buffer, indices)
        # update network parameters
        result = self.learn(batch, **kwargs)
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False
        return result

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        sigma = 0.1
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        model_ = self._actor if model == "actor" else self._target_actor
        mu, hidden = model_(obs_), None
        if self.training:
            # 重参数化采样
            noise = torch.randn_like(mu)  # 从标准正态分布采样
            acts = mu + sigma * noise  # 可微操作
        else:
            acts = mu
        dist = Independent(Normal(mu, sigma), 1)
        acts = torch.clamp(acts, max=1, min=-1)
        return Batch(logits=mu, act=acts, state=hidden, dist=dist)

    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        # acts_ = to_torch(batch.act[:, np.newaxis], device=self._device, dtype=torch.long)
        acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)
        target_q = batch.returns
        # target_q = to_torch(batch.rew, device=self._device, dtype=torch.float32).unsqueeze(1)
        current_q1, current_q2 = self._critic(obs_, acts_)
        critic_loss = F.mse_loss(current_q1, target_q) \
                      + F.mse_loss(current_q2, target_q)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        
        return critic_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        next_result = self.forward(batch)
        dist = next_result.dist
        # entropy = dist.entropy()
        # with torch.no_grad():
        #     q = self._critic.q_min(obs_, next_result.act)
        next_result.log_prob = dist.log_prob(next_result.act)
        q = self._critic.q_min(obs_, next_result.logits)
        pg_loss = (self._alpha * next_result.log_prob.unsqueeze(1) - q).mean()
        
        self._actor_optim.zero_grad()
        pg_loss.backward()
        self._actor_optim.step()
        return pg_loss, next_result

    def _update_targets(self):
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        # update critic network
        critic_loss = self._update_critic(batch)
        # update actor network
        pg_loss, next_result = self._update_policy(batch, update=False)
        
        if self._is_auto_alpha:
            log_prob = next_result.log_prob.detach() + self._target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
            self._alpha = torch.clamp(self._alpha, max=1, min=0)
        # update target networks
        self._update_targets()
        return {
            'loss/critic': critic_loss.item(),
            'pg_loss': pg_loss.item(),
            'alpha':self._alpha.item(),
            'loss/alpha':alpha_loss.item()
        }
