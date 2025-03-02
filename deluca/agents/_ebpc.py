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
jax.config.update("jax_debug_nans", True)
import scipy.optimize
import scipy.linalg


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

# @partial(jax.jit, static_argnames=['jac', 'hes'])
# def newton_iterate(x, jac, hes):
#     jac_val = jac(x)
#     hes_val = hes(x)
#     hes_inv_jac_val = jlinalg.inv(hes_val) @ jac_val
#     return (x - hes_inv_jac_val/1.1)
#     #lam = jnp.sum(hes_inv_jac_val * jac_val)**.5
#     #return jax.lax.cond(lam < 1, lambda x: (x-hes_inv_jac_val), lambda x: (x - hes_inv_jac_val/(1+lam)), x)

# @partial(jax.jit, static_argnames=['jac', 'hes'])
# def newton_step_minimum(jac, hes, start_point):
#     """
#     Description: fast minimizer for self-concordant functions
#     """
#     # x = start_point
#     # while jnp.linalg.norm(jac(x))>1e-8:
#     #     x = newton_iterate(x, jac, hes)
#     #     print("iter:", x)
#     # print(hash(jac), hash(hes))
#     x = jax.lax.while_loop(lambda x: jnp.linalg.norm(jac(x))>1e-6, lambda x: newton_iterate(x, jac, hes), start_point)
#     return x


@partial(jax.jit, static_argnames=['jac', 'hes'])
def newton_iterate_specific(x, jac, hes, old_g_sum, old_M_sum, t):
    jac_val = jac(x, old_g_sum, old_M_sum, t)
    hes_val = hes(x, t)
    hes_inv_jac_val = jlinalg.solve(hes_val, jac_val)
    lam = jnp.dot(hes_inv_jac_val, jac_val) ** .5
    return (x-hes_inv_jac_val/(1.1+lam), jnp.linalg.norm(hes_inv_jac_val/(1.1+lam)))
    

@partial(jax.jit, static_argnames=['jac', 'hes'])
def newton_step_minimum_specific(jac, hes, old_g_sum, old_M_sum, t, start_point):
    # x = (start_point, 1e10)
    # niter = 0
    # while x[1] >= 1e-7:
    #     x = newton_iterate_specific(x[0], jac, hes, old_g_sum, old_M_sum, t)
    #     print("Iter!", x[1])
    #     print(x[0])
    #     niter += 1
    #     if niter > 30:
    #         raise ArithmeticError
    x = jax.lax.while_loop(lambda x: x[1] >= 1e-7, lambda x: newton_iterate_specific(x[0], jac, hes, old_g_sum, old_M_sum, t), (start_point, 1e10))
    return x[0]
    

def stable_sqrtm(mat):  # Needed since jax sqrt appears to have issues with poor conditioning. NOTE: results in substantial slowdown. Any way to go back to JAX?
    return jnp.array(scipy.linalg.sqrtm(np.array(mat)))
    # min_elt = jnp.min(jnp.abs(mat))
    # min_elt = jax.lax.cond(min_elt > 1e-10, lambda x: x, lambda x: 1.0, min_elt)
    # return jlinalg.sqrtm(mat / min_elt) * (min_elt ** 0.5)



class EBPC(Agent):
    def __init__(
        self,
        A: jnp.ndarray = None,  # Linear system matrices. Will override Markov operator if set.
        B: jnp.ndarray = None,
        C: jnp.ndarray = None,  # If none, assumes full observation
        Q: jnp.ndarray = None,  # Cost matrices
        R: jnp.ndarray = None,  # Cost matrices
        T: int = 10000,  # Time horizon
        H: int = 5,  # Memory duration
        cost_bound: Real = 1,  # Maximum cost
        sigma: Real = 0.01,  # Strong-convexity guarantee
        beta: Real = 100,  # Smoothness guarantee
        rad: Real = 100,  # Radius of controller set
        L: Real = 10,  # Lipschitz constant
        eta_mul: Real = 1,  # Used for practical tuning  
        grad_mul: Real = 1,  # used for practical tuning
        use_K: bool = False,
        K: jnp.ndarray = None,
        random_key: jax.random.PRNGKey = None,

    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        """
        self.A = A
        self.B = B
        self.C = C
        if self.C is None:
            self.C = jnp.identity(self.A.shape[0])
        else:
            if use_K:
                raise AssertionError("Requires C=None if use_K is specified!")
        self.use_K = use_K
        if use_K:
            self.K = K if K is not None else LQR(self.A, self.B, Q, R).K
        self.d_out, self.d_state, self.d_action = self.C.shape[0], self.C.shape[1], self.B.shape[1]  # State & Action Dimensions
        self.H = H
        def eta_min_func(eta):
            return ((beta*H*np.log(T))/(2*eta*sigma)
                     + (16*np.sqrt(eta*T)*self.d_state*L*cost_bound*H**3)/np.sqrt(sigma)
                     + 32*eta*self.d_state**2*cost_bound**2*H**7*T
                     + (2*np.log(T))/eta
                     + (16*np.sqrt(eta*T)*beta*self.d_state*cost_bound*(2*rad)*H**4)/np.sqrt(sigma))
        self.eta = eta_mul * scipy.optimize.minimize(eta_min_func, 1/np.sqrt(T), bounds=((1e-8/T, None),)).x[0]
        print("eta:", self.eta)
        self.t = 0  # Time Counter (for decaying learning rate)
        self.rad = rad

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
            return -jnp.log(1 - jnp.sum(jnp.square(flat_M)) / (self.rad**2))

        # Compiling this in place is really bad, since it ends up baking all of the constants in place.
        def min_func(flat_M):  # This is a rewrite of the function minimized to obtain M in terms of the sums of historic Ms and gs.
            return (self.eta * jnp.sum(jnp.ravel(self.old_g_sum) * flat_M)
                    + self.eta * (self.sigma/2.0) * (self.t-self.H+1) * jnp.sum(jnp.square(flat_M - jnp.ravel(self.old_M_sum/(self.t-self.H+1))))
                    + regularizer(flat_M)
                   )
        
        self.regularizer = regularizer
        self.min_func = min_func

        self.jac_regularizer = jax.jit(jax.jacobian(regularizer))
        self.hes_regularizer = jax.jit(jax.hessian(regularizer))

        @jax.jit
        def jac_min_func(flat_M, old_g_sum, old_M_sum, t):  # Manually computed
            return (self.eta * jnp.ravel(old_g_sum)
                    + self.eta * self.sigma * (t-self.H+1) * (flat_M - jnp.ravel(old_M_sum/(t-self.H+1)))
                    + jax.jit(self.jac_regularizer)(flat_M)
                    )
        
        @jax.jit
        def hes_min_func(flat_M, t):  # Likewise
            return self.eta * self.sigma * (t-self.H+1) * jnp.identity(flat_M.shape[0]) + jax.jit(self.hes_regularizer)(flat_M)
        

        self.jac_min_func = jac_min_func
        self.hes_min_func = hes_min_func

        self.key = random_key
        if self.key is None:
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
            self.M = jnp.reshape(newton_step_minimum_specific(self.jac_min_func,
                                                              self.hes_min_func, 
                                                              self.old_g_sum, 
                                                              self.old_M_sum, 
                                                              self.t, 
                                                              jnp.ravel(self.M)), 
                                 (self.H, self.d_action, self.d_out))
        else:
            self.M = jnp.zeros((self.H, self.d_action, self.d_out))
        if jnp.sum(jnp.square(self.M)) / (self.rad**2) > 1-1e-6:  # make sure we stay inside the boundary -- floating-point causes problems here
            self.M = self.M * (1-1e-6) / (jnp.sum(jnp.square(self.M)) / (self.rad**2))**.5
        self.g_buffer = roll_and_set_last(self.g_buffer, g)
        self.M_buffer = roll_and_set_last(self.M_buffer, self.M)
        hessian_sum = self.hes_regularizer(jnp.ravel(self.M)) + self.eta * self.sigma * (self.t+1) * jnp.identity(np.prod(self.M.shape))

        A_inv = jnp.reshape(jnp.real(stable_sqrtm(hessian_sum)), self.M.shape + self.M.shape)

        self.A_inv = roll_and_set_last(self.A_inv, A_inv)

        eps = generate_uniform_seeded(self.M.shape, temp_key)

        self.eps = roll_and_set_last(self.eps, eps)
        self.tilde_M = self.M + jnp.reshape(jnp.linalg.solve(jnp.reshape(A_inv, (np.prod(self.M.shape),)*2), jnp.ravel(eps)), self.M.shape)
    def get_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: get action from state.

        Args:
            state (jnp.ndarray):

        Returns:
            jnp.ndarray
        """
        new_u = jnp.tensordot(self.tilde_M, self.y_nat[::-1], ((0, 2,), (0, 1,)))  # Todo: check this and other tensordots.
        if self.use_K:
            new_u = new_u - (self.K @ state)[:, 0]
        self.u = roll_and_set_last(self.u, new_u)
        self.x_unnat = self.A.dot(self.x_unnat) + self.B.dot(new_u)
        # print("YNAT:", self.y_nat)
        # print("Control:", new_u)
        return new_u[:, jnp.newaxis]