# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""deluca.agents._bpc"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as random
from jax import grad
from jax import jit

from deluca.agents._lqr import LQR
from deluca.agents.core import Agent

def generate_uniform(shape, norm=1.00):
            v = random.normal(size=shape)
            v = norm * v / np.linalg.norm(v)
            v = np.array(v)
            return v

def onetwo_norm(tens):
     return np.sum(np.linalg.norm(tens, axis=(1, 2)))

class BPC(Agent):
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        Q: jnp.ndarray = None,
        R: jnp.ndarray = None,
        K: jnp.ndarray = None,
        start_time: int = 0,
        H: int = 5,
        lr_scale: Real = 0.005,
        decay: bool = False,
        delta: Real = 0.01,
        use_K: bool = False,  # Determines whether we augment with a stabilizing controller. 
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (jnp.ndarray): system dynamics
            B (jnp.ndarray): system dynamics
            Q (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            R (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            K (jnp.ndarray): Starting policy (optional). Defaults to LQR gain.
            start_time (int):
            H (postive int): history of the controller
            lr_scale (Real):
            decay (boolean):
        """

        self.d_state, self.d_action = B.shape  # State & Action Dimensions

        self.A, self.B = A, B  # System Dynamics

        self.t = 0  # Time Counter (for decaying learning rate)

        self.H = H

        self.lr_scale, self.decay = lr_scale, decay

        self.delta = delta

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        # TODO: need to address problem of LQR with jax.lax.scan
        self.K = K if K is not None else LQR(self.A, self.B, Q, R).K

        self.tilde_M = self.M = self.delta * generate_uniform((H, self.d_action, self.d_state))

        # Past H noises ordered increasing in time
        self.noise_history = jnp.zeros((H, self.d_state, 1))

        # past state and past action
        self.state, self.action = jnp.zeros((self.d_state, 1)), jnp.zeros((self.d_action, 1))

        self.eps = generate_uniform((H, H, self.d_action, self.d_state))
        self.eps_bias = generate_uniform((H, self.d_action, 1))

        def grad(M, noise_history, cost):
            return cost * jnp.sum(self.eps, axis = 0)

        self.grad = grad

        self.use_K = use_K

    def __call__(self,
                state: jnp.ndarray,
                cost: Real
                ) -> jnp.ndarray:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (jnp.ndarray): current state

        Returns:
           jnp.ndarray: action to take
        """

        action = self.get_action(state)
        self.update(state, action, cost)
        return action

    def update(self,
            state: jnp.ndarray,
            action:jnp.ndarray,
            cost: Real
            ) -> None:
        """
        Description: update agent internal state.

        Args:
            state (jnp.ndarray): current state
            action (jnp.ndarray): action taken
            cost (Real): scalar cost received

        Returns:
            None
        """
        noise = state - self.A @ self.state - self.B @ action
        self.noise_history = self.noise_history.at[0].set(noise)
        self.noise_history = jnp.roll(self.noise_history, -1, axis=0)

        lr = self.lr_scale
        lr *= (1/ (self.t**(3/4)+1)) if self.decay else 1

        delta_M = self.grad(self.M, self.noise_history, cost)
        self.M -= lr * delta_M
        if onetwo_norm(self.M) > (1-2*self.delta):
            self.M *= .8 / onetwo_norm(self.M)

        self.eps[0] = generate_uniform((self.H, self.d_action, self.d_state))
        self.eps = np.roll(self.eps, -1, axis = 0)
        self.tilde_M = self.M + self.delta * self.eps[-1]

        # update state
        self.state = state

        self.t += 1

    def get_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: get action from state.

        Args:
            state (jnp.ndarray):

        Returns:
            jnp.ndarray
        """
        return (-self.K @ state if self.use_K else 0) + jnp.tensordot(self.tilde_M, self.noise_history, axes=([0, 2], [0, 1]))


