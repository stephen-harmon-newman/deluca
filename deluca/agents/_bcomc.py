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
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
import numpy as np
import numpy.random as random
from jax import grad
from jax import jit
from jax.scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)
import scipy.optimize


from deluca.agents._lqr import LQR
from deluca.agents.core import Agent

def generate_uniform_seeded(shape, key, norm=1.00):
    v = jax.random.normal(key, shape)
    v = norm * v / np.linalg.norm(v)
    v = np.array(v)
    return v

@jax.jit
def roll_and_set_last(arr, val):
    return jnp.roll(arr.at[0].set(val), -1, 0)

def truncated_markov_operator(A, B, C, H):  
    """
    Description: given the system matrices A, B, C,
    generate the H-step truncated Markov operator
    """
    
    G = jnp.zeros(((H,) + B.shape))
    for i in range(H):
        G = G.at[i].set(C @ jnp.linalg.matrix_power(A, i) @ B)
    return G

@partial(jax.jit, static_argnames=['jac', 'hes'])
def newton_iterate(x, jac, hes):
    jac_val = jac(x)
    hes_val = hes(x)
    hes_inv_jac_val = jlinalg.inv(hes_val) @ jac_val
    return (x - hes_inv_jac_val/1.1)
    #lam = jnp.sum(hes_inv_jac_val * jac_val)**.5
    #return jax.lax.cond(lam < 1, lambda x: (x-hes_inv_jac_val), lambda x: (x - hes_inv_jac_val/(1+lam)), x)

@partial(jax.jit, static_argnames=['jac', 'hes'])
def newton_step_minimum(jac, hes, start_point):
    """
    Description: fast minimizer for self-concordant functions
    """
    # x = start_point
    # while jnp.linalg.norm(jac(x))>1e-8:
    #     x = newton_iterate(x, jac, hes)
    #     print("iter:", x)
    # print(hash(jac), hash(hes))
    x = jax.lax.while_loop(lambda x: jnp.linalg.norm(jac(x))>1e-6, lambda x: newton_iterate(x, jac, hes), start_point)
    return x


@partial(jax.jit, static_argnames=['jac', 'hes'])
def newton_iterate_specific(x, jac, hes, old_g_sum, old_M_sum):
    jac_val = jac(x, old_g_sum, old_M_sum)
    hes_val = hes(x)
    hes_inv_jac_val = jlinalg.solve(hes_val, jac_val)
    lam = jnp.dot(hes_inv_jac_val, jac_val) ** .5
    return (x-hes_inv_jac_val/ (2+2*lam), jnp.sum(jnp.square(hes_inv_jac_val))/(1+lam))
    

@partial(jax.jit, static_argnames=['jac', 'hes'])
def newton_step_minimum_specific(jac, hes, old_g_sum, old_M_sum, start_point):
    x = jax.lax.while_loop(lambda x: x[1] >= 1e-10, lambda x: newton_iterate_specific(x[0], jac, hes, old_g_sum, old_M_sum), (start_point, 1e10))
    return x[0]
    




class BCOMC(Agent):
    def __init__(
        self,
        A: jnp.ndarray = None,  # Linear system matrices. Will override Markov operator if set.
        B: jnp.ndarray = None,
        C: jnp.ndarray = None,
        T: int = 10000,  # Time horizon
        H: int = 5,  # Memory duration
        cost_bound: Real = 1,  # Maximum cost
        sigma: Real = 0.01,  # Strong-convexity guarantee
        beta: Real = 100,  # Smoothness guarantee
        R: Real = 100,  # Radius of controller set
        L: Real = 10,  # Lipschitz constant
        eta_mul: Real = 1,  # Used for practical tuning  
        grad_mul: Real = 1,  # used for practical tuning  
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        """
        self.A = A
        self.B = B
        self.C = C
        self.d_out, self.d_state, self.d_action = self.C.shape[0], self.C.shape[1], self.B.shape[1]  # State & Action Dimensions
        self.H = H
        def eta_min_func(eta):
            return ((beta*H*np.log(T))/(2*eta*sigma)
                     + (16*np.sqrt(eta*T)*self.d_state*L*cost_bound*H**3)/np.sqrt(sigma)
                     + 32*eta*self.d_state**2*cost_bound**2*H**7*T
                     + (2*np.log(T))/eta
                     + (16*np.sqrt(eta*T)*beta*self.d_state*cost_bound*(2*R)*H**4)/np.sqrt(sigma))
        self.eta = eta_mul * scipy.optimize.minimize(eta_min_func, 1/np.sqrt(T), bounds=((1e-8/T, None),)).x[0]
        print("eta:", self.eta)
        self.t = 0  # Time Counter (for decaying learning rate)
        self.R = R

        self.M = jnp.zeros((H, self.d_action, self.d_out))  # We don't need history for this
        self.tilde_M = jnp.zeros((H, self.d_action, self.d_out))  # Or this
        self.eps = jnp.zeros((H, H, self.d_action, self.d_out))  # But we do for this and the rest
        self.y_nat = jnp.zeros((H, self.d_out))
        self.A_inv = jnp.zeros((H, H, self.d_action, self.d_out, H, self.d_action, self.d_out))
        self.u = jnp.zeros((H, self.d_action))

        self.x_unnat = jnp.zeros((self.d_out,))


        for i in range(H):
            self.A_inv = self.A_inv.at[i].set(jnp.reshape(jnp.identity(np.prod(self.M.shape)), self.M.shape * 2))

        """
        Note: for the horizon-H arrays, we're going to follow the convention that the most recently added things are last in the array. 
        This is how _bpc.py does it, anyway.
        """


        """
        We need to store a condensed version of the g_t s and M_t s so that we can efficiently do the minimum computation. 
        """
        self.old_g_sum = jnp.zeros((H, self.d_action, self.d_out))
        self.old_M_sum = jnp.zeros((H, self.d_action, self.d_out))
        self.g_buffer = jnp.zeros((H, H, self.d_action, self.d_out))
        self.M_buffer = jnp.zeros((H, H, self.d_action, self.d_out))


        @jax.jit
        def regularizer(flat_M):
            return -jnp.log(1 - jnp.sum(jnp.square(flat_M)) / (self.R**2))

        # Compiling this in place is really bad, since it ends up baking all of the constants in place.
        def min_func(flat_M):
            return (self.eta * jnp.sum(jnp.ravel(self.old_g_sum) * flat_M)
                    + self.eta * (self.sigma/2.0) * jnp.sum(jnp.square(flat_M - jnp.ravel(self.old_M_sum)))
                    + regularizer(flat_M)
                   )
        
        self.regularizer = regularizer
        self.min_func = min_func

        self.jac_regularizer = jax.jit(jax.jacobian(regularizer))
        self.hes_regularizer = jax.jit(jax.hessian(regularizer))

        @jax.jit
        def jac_min_func(flat_M, old_g_sum, old_M_sum):
            return (self.eta * jnp.ravel(old_g_sum)
                    + self.eta * self.sigma * jnp.sum(flat_M - jnp.ravel(old_M_sum))
                    + self.jac_regularizer(flat_M)
                    )
        
        @jax.jit
        def hes_min_func(flat_M):
            return self.eta * self.sigma * jnp.identity(flat_M.shape[0]) + self.hes_regularizer(flat_M)
        

        self.jac_min_func = jac_min_func
        self.hes_min_func = hes_min_func

        self.key = jax.random.PRNGKey(42)

        self.grad_mul = grad_mul

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

        self.update(state, cost)
        action = self.get_action(state)
        return action

    def update(self,
            state: jnp.ndarray,
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

        self.key, temp_key = jax.random.split(self.key)

        self.t = self.t + 1
        new_y_nat = state[:, 0] - self.C.dot(self.x_unnat)
        #print("Measured State:", state[:, 0])
        #print("Computed y_nat:", new_y_nat)
        self.y_nat = roll_and_set_last(self.y_nat, new_y_nat)
        """
        We do this half of the update before the g, M calculation to keep the indices right for M calculation.
        Roll occurs after.
        """
        self.old_g_sum = self.old_g_sum + self.g_buffer[0]
        self.old_M_sum = self.old_M_sum + self.M_buffer[0]
        g = self.grad_mul * self.d_action * self.d_out * self.H**2 * cost * jnp.tensordot(self.A_inv, self.eps, ((0,4,5,6), (0,1,2,3)))
        if self.t >= self.H:
            # self.M = jnp.reshape(minimize(self.min_func, jnp.ravel(jnp.zeros(self.M.shape)), method="BFGS").x, self.M.shape)
            self.M = jnp.reshape(newton_step_minimum_specific(self.jac_min_func, self.hes_min_func, self.old_g_sum, self.old_M_sum, jnp.ravel(jnp.zeros(self.M.shape))), (self.H, self.d_action, self.d_out))
        else:
            self.M = jnp.zeros((self.H, self.d_action, self.d_out))
        
        self.g_buffer = roll_and_set_last(self.g_buffer, g)
        self.M_buffer = roll_and_set_last(self.M_buffer, self.M)
        hessian_sum = self.hes_regularizer(jnp.ravel(self.M)) + self.eta * self.sigma * (self.t+1) * jnp.identity(np.prod(self.M.shape))
        A_inv = jnp.reshape(jnp.real(jlinalg.sqrtm(hessian_sum)), self.M.shape + self.M.shape)

        self.A_inv = roll_and_set_last(self.A_inv, A_inv)

        eps = generate_uniform_seeded(self.M.shape, temp_key)
        self.eps = roll_and_set_last(self.eps, eps)
        self.tilde_M = self.M + jnp.reshape(jnp.linalg.solve(jnp.reshape(A_inv, (np.prod(self.M.shape),)*2), jnp.ravel(eps)), self.M.shape)
        # self.M + jnp.tensordot(jnp.reshape(jlinalg.inv(jnp.reshape(A_inv, (np.prod(self.M.shape),)*2)), self.M.shape*2), eps, ((3,4,5),(0,1,2)))
        if self.t % 100 == 0:
            print("g:", g)
            print("M:", self.M)
            print("~M:", self.tilde_M)
            print("y_nat:", self.y_nat)
    def get_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: get action from state.

        Args:
            state (jnp.ndarray):

        Returns:
            jnp.ndarray
        """
        new_u = jnp.tensordot(self.tilde_M, self.y_nat[::-1], ((0, 2,), (0, 1,)))  # Todo: check this and other tensordots.
        self.u = roll_and_set_last(self.u, new_u)
        self.x_unnat = self.A.dot(self.x_unnat) + self.B.dot(new_u)
        # print("YNAT:", self.y_nat)
        # print("Control:", new_u)
        return new_u[:, jnp.newaxis]
