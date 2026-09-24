"""PPO for the centralized formation controller (CleanRL-style, single file).

The actor is NOT defined here. It lives in :mod:`formation_rl.actor` and is
reached only through ``actor.act`` and ``actor(obs)``, so replacing it with a
spiking network is a change to that module alone. What lives here is the
training loop, the critic (which is discarded after training and therefore has
no architecture constraints), and the Gaussian exploration head.

Exploration: the actor is deterministic and tanh-bounded, so PPO samples from a
Normal centred on its output with a learned state-independent log-std, and the
log-std is annealed by the usual entropy term. Acting greedily at evaluation
means the deployed policy is exactly ``actor.act``, with no sampling on the
robot -- which matters, because a spiking actor has nowhere to put a Gaussian.

Defaults are recorded in the README. They are ordinary PPO settings, tuned only
enough to beat the analytic baseline on this task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from formation_core.contract import ACTION_DIM, OBS_DIM

from .actor import DEFAULT_HIDDEN, MlpActor, mlp


@dataclass
class PPOConfig:
    """Hyperparameters. Documented in the README; changing them is a config edit."""

    total_steps: int = 600_000
    #: Parallel environments, each a full episode of the task on its own seed.
    num_envs: int = 8
    #: Steps collected per environment per update -> batch = this * num_envs.
    rollout_steps: int = 256
    minibatches: int = 8
    update_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    #: Initial log-std of the exploration Gaussian. The action is a normalized
    #: velocity, so 0.4 rad/s-ish of initial jitter is a reasonable start.
    init_log_std: float = -1.0
    hidden: tuple = field(default_factory=lambda: tuple(DEFAULT_HIDDEN))
    seed: int = 0
    #: Episode seeds cycle through this many values, so the policy cannot
    #: overfit one realisation of the leader's process noise.
    train_seeds: int = 64

    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self):
        return max(self.batch_size // self.minibatches, 1)

    def to_dict(self):
        out = asdict(self)
        out['hidden'] = list(self.hidden)
        return out


class Critic(nn.Module):
    """Value head. Training-only, so it is under no conversion constraints."""

    def __init__(self, obs_dim=OBS_DIM, hidden=(64, 64)):
        super().__init__()
        self.net = mlp((obs_dim, *hidden, 1), activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Couples the swappable actor with a throwaway critic and an std head."""

    def __init__(self, actor=None, hidden=DEFAULT_HIDDEN, init_log_std=-1.0):
        super().__init__()
        self.actor = actor if actor is not None else MlpActor(hidden=hidden)
        self.critic = Critic(hidden=tuple(hidden))
        # State-independent std: one parameter per action dimension. The actor
        # itself stays deterministic, which is what gets deployed.
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), float(init_log_std)))

    def value(self, obs):
        return self.critic(obs)

    def distribution(self, obs):
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def act_and_value(self, obs, action=None):
        dist = self.distribution(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.critic(obs)


class VectorEnv:
    """Minimal synchronous vector env.

    Gymnasium ships one, but ours has to do something its wrappers do not: give
    each sub-environment a DIFFERENT episode seed on every reset, cycling
    through ``train_seeds`` values, so the policy sees many realisations of the
    leader's process noise rather than memorising one.
    """

    def __init__(self, config, num_envs, seeds):
        from .gym_env import FormationGymEnv

        self.envs = [FormationGymEnv(config) for _ in range(num_envs)]
        self.seeds = list(seeds)
        self._cursor = list(range(num_envs))
        self.num_envs = num_envs

    def _next_seed(self, index):
        seed = self.seeds[self._cursor[index] % len(self.seeds)]
        self._cursor[index] += self.num_envs
        return seed

    def reset(self):
        obs = [env.reset(seed=self._next_seed(i))[0]
               for i, env in enumerate(self.envs)]
        return np.stack(obs)

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                # Autoreset, as Gymnasium's vector envs do: the value of the
                # final state is bootstrapped before the reset, below.
                info = dict(info, final_observation=obs, terminated=terminated)
                obs, _ = env.reset(seed=self._next_seed(i))
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(observations), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.float32), infos


def train(config, ppo=None, progress=None, device='cpu'):
    """Run PPO and return the trained :class:`MlpActor` plus a history list."""
    ppo = ppo or PPOConfig()
    torch.manual_seed(ppo.seed)
    np.random.seed(ppo.seed)

    seeds = [ppo.seed * 1000 + i for i in range(ppo.train_seeds)]
    envs = VectorEnv(config, ppo.num_envs, seeds)
    agent = ActorCritic(hidden=ppo.hidden, init_log_std=ppo.init_log_std).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=ppo.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((ppo.rollout_steps, ppo.num_envs, OBS_DIM), device=device)
    act_buf = torch.zeros((ppo.rollout_steps, ppo.num_envs, ACTION_DIM), device=device)
    logp_buf = torch.zeros((ppo.rollout_steps, ppo.num_envs), device=device)
    rew_buf = torch.zeros((ppo.rollout_steps, ppo.num_envs), device=device)
    done_buf = torch.zeros((ppo.rollout_steps, ppo.num_envs), device=device)
    val_buf = torch.zeros((ppo.rollout_steps, ppo.num_envs), device=device)

    next_obs = torch.as_tensor(envs.reset(), dtype=torch.float32, device=device)
    next_done = torch.zeros(ppo.num_envs, device=device)
    updates = max(ppo.total_steps // ppo.batch_size, 1)
    history = []
    started = time.monotonic()
    returns_window = []

    for update in range(1, updates + 1):
        # Linear learning-rate decay, the CleanRL default.
        for group in optimizer.param_groups:
            group['lr'] = ppo.learning_rate * (1.0 - (update - 1.0) / updates)

        for step in range(ppo.rollout_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, log_prob, _, value = agent.act_and_value(next_obs)
            act_buf[step] = action
            logp_buf[step] = log_prob
            val_buf[step] = value

            obs, reward, done, infos = envs.step(
                np.clip(action.cpu().numpy(), -1.0, 1.0))
            rew_buf[step] = torch.as_tensor(reward, device=device)
            next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, device=device)
            for info in infos:
                if 'final_observation' in info:
                    returns_window.append(float(info.get('reward', 0.0)))

        with torch.no_grad():
            next_value = agent.value(next_obs)
            advantages = torch.zeros_like(rew_buf)
            last_gae = 0.0
            for t in reversed(range(ppo.rollout_steps)):
                if t == ppo.rollout_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    values_next = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    values_next = val_buf[t + 1]
                delta = (rew_buf[t] + ppo.gamma * values_next * next_nonterminal
                         - val_buf[t])
                last_gae = (delta + ppo.gamma * ppo.gae_lambda
                            * next_nonterminal * last_gae)
                advantages[t] = last_gae
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, OBS_DIM)
        b_act = act_buf.reshape(-1, ACTION_DIM)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        indices = np.arange(ppo.batch_size)
        for _ in range(ppo.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, ppo.batch_size, ppo.minibatch_size):
                batch = indices[start:start + ppo.minibatch_size]
                _, new_logp, entropy, new_value = agent.act_and_value(
                    b_obs[batch], b_act[batch])
                ratio = (new_logp - b_logp[batch]).exp()

                adv = b_adv[batch]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - ppo.clip_coef, 1 + ppo.clip_coef),
                ).mean()
                value_loss = 0.5 * ((new_value - b_ret[batch]) ** 2).mean()
                loss = (policy_loss
                        + ppo.value_coef * value_loss
                        - ppo.entropy_coef * entropy.mean())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo.max_grad_norm)
                optimizer.step()

        record = {
            'update': update,
            'steps': update * ppo.batch_size,
            'mean_reward': float(rew_buf.mean().item()),
            'value_loss': float(value_loss.item()),
            'policy_loss': float(policy_loss.item()),
            'log_std': float(agent.log_std.mean().item()),
            'elapsed': time.monotonic() - started,
        }
        history.append(record)
        if progress is not None:
            progress(record)

    agent.actor.eval()
    return agent.actor, history
