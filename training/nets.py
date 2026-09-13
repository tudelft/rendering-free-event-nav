"""Actor-critic for the PPO arm.

ACTOR is deploy-shaped for the N6 Neural-ART NPU: conv/relu/dense/concat ONLY
(no norm layers, no silu/softmax/tanh inside the graph), linear output head.
The policy is a TANH-SQUASHED Gaussian, but the squash lives OUTSIDE the
network: training samples u ~ N(mean, std) and applies a = tanh(u) in
train_ppo.py (with the log-det correction in logp_fn); eval/deploy apply
a = tanh(mean) — on the N6 that is two scalar ops on the M55 after the single
NPU pass. Input frames uint8; the (x-128)/32 affine is the first op so it
folds into conv0's int8 scale at quantization.

CRITIC is training-only and privileged (sees priv[15]) — the asymmetric
actor-critic version of the Informed trick.
"""
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

ORTH = nn.initializers.orthogonal
def _conv(f, k, s_):
    return nn.Conv(f, (k, k), strides=s_, kernel_init=ORTH(np.sqrt(2)))
def _dense(f, scale=np.sqrt(2)):
    return nn.Dense(f, kernel_init=ORTH(scale))


# Trunk/head widths. Default = the DEPLOYED 1.2MB architecture (16/32/64 conv,
# 256/128 heads) — the 32/128/128 widening that slipped in with the deploy-chain
# commit showed no measurable gain (renderdr 16/20 == headpen 16/20) at 4x the
# parameters. Override for experiments via env PPO_WIDTHS="c1,c2,c3,d1,d2",
# e.g. PPO_WIDTHS="8,16,32,256,128" for the half-channel variant.
import os
_W = tuple(int(x) for x in os.environ.get('PPO_WIDTHS', '16,32,64,256,128').split(','))
CW, DW = _W[:3], _W[3:5]


class ConvTrunk(nn.Module):
    @nn.compact
    def __call__(self, img):                       # [B,H,H,C] uint8
        x = (img.astype(jnp.float32) - 128.0) / 32.0
        x = nn.relu(_conv(CW[0], 5, 2)(x))
        x = nn.relu(_conv(CW[1], 3, 2)(x))
        x = nn.relu(_conv(CW[2], 3, 2)(x))   # H/8 x H/8 x CW[2]
        return x.reshape((x.shape[0], -1))


class Actor(nn.Module):
    adim: int = 2

    @nn.compact
    def __call__(self, img, vec):
        z = ConvTrunk()(img)
        v = nn.relu(_dense(64)(vec))
        h = nn.relu(_dense(DW[0])(jnp.concatenate([z, v], -1)))
        h = nn.relu(_dense(DW[1])(h))
        mean = nn.Dense(self.adim, kernel_init=ORTH(0.01))(h)   # small-init head
        log_std = self.param('log_std', nn.initializers.constant(-0.5),
                             (self.adim,))
        log_std = jnp.clip(log_std, -2.5, 0.5)      # exploration floor + guard
        return mean, jnp.broadcast_to(log_std, mean.shape)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, img, vec, priv):
        z = ConvTrunk()(img)
        v = nn.relu(_dense(64)(jnp.concatenate([vec, priv], -1)))
        h = nn.relu(_dense(DW[0])(jnp.concatenate([z, v], -1)))
        h = nn.relu(_dense(DW[1])(h))
        return nn.Dense(1, kernel_init=ORTH(1.0))(h)[..., 0]
