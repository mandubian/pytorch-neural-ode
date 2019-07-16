# Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
import torch
import torch.nn as nn

from torchdiffeq._impl.misc import (
    _scaled_dot_product, _convert_to_tensor, _is_finite, _is_iterable,
    _optimal_step_size, _compute_error_ratio
)
from torchdiffeq._impl.solvers import AdaptiveStepsizeODESolver
from torchdiffeq._impl.interp import _interp_fit, _interp_evaluate
from torchdiffeq._impl.rk_common import _RungeKuttaState, _ButcherTableau
from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.misc import _flatten, _flatten_convert_none_to_zeros, _decreasing, _norm

from odeint_ext.odeint_ext_misc import _select_initial_step


_DORMAND_PRINCE_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=[1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.],
    beta=[
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    c_sol=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    c_error=[
        35 / 384 - 1951 / 21600,
        0,
        500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720,
        -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300,
        -1. / 60.,
    ],
)

DPS_C_MID = [
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
    187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
]


def _interp_fit_dopri5(y0, y1, k, dt, tableau=_DORMAND_PRINCE_SHAMPINE_TABLEAU):
    """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
    dt = dt.type_as(y0[0])
    y_mid = tuple(y0_ + _scaled_dot_product(dt, DPS_C_MID, k_) for y0_, k_ in zip(y0, k))
    f0 = tuple(k_[0] for k_ in k)
    f1 = tuple(k_[-1] for k_ in k)
    return _interp_fit(y0, y1, y_mid, f0, f1, dt)

def _abs_square(x):
    return torch.mul(x, x)


def _ta_append(list_of_tensors, value):
    """Append a value to the end of a list of PyTorch tensors."""
    list_of_tensors.append(value)
    return list_of_tensors



def _runge_kutta_step(func, y0, f0, t0, dt, tableau, **unused_kwargs):
    """Take an arbitrary Runge-Kutta step and estimate error.

    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative
            of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        tableau: optional _ButcherTableau describing how to take the Runge-Kutta
            step.
        name: optional name for the operation.

    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """
    dtype = y0[0].dtype
    device = y0[0].device

    t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
    dt = _convert_to_tensor(dt, dtype=dtype, device=device)
    
#    if not torch.is_tensor(f0[0]):
#        f0, *r = f0
#        f0, *r_keep = f0
#        f0 = f0, *r
#        k_keep = [r_keep]
#    else:
#        k_keep = []
    k = tuple(map(lambda x: [x], f0))

    for alpha_i, beta_i in zip(tableau.alpha, tableau.beta):
        ti = t0 + alpha_i * dt
        yi = tuple(y0_ + _scaled_dot_product(dt, beta_i, k_) for y0_, k_ in zip(y0, k))
        fi = func(ti, yi, **unused_kwargs)
        #if not torch.is_tensor(fi[0]):
        #    fi, *r = fi
        #    fi, *r_keep = fi
        #    fi = fi, *r            
        #    k_keep.append(r_keep)
        tuple(k_.append(f_) for k_, f_ in zip(k, fi))

    if not (tableau.c_sol[-1] == 0 and tableau.c_sol[:-1] == tableau.beta[-1]):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = tuple(y0_ + _scaled_dot_product(dt, tableau.c_sol, k_) for y0_, k_ in zip(y0, k))

    y1 = yi
    #if len(k_keep) > 0:
    #    f1 = tuple((k_[-1], *k_keep_) for k_, k_keep_ in zip(k, k_keep))
    #else:
    f1 = tuple(k_[-1] for k_ in k)
    y1_error = tuple(_scaled_dot_product(dt, tableau.c_error, k_) for k_ in k)
    return (y1, f1, y1_error, k)


class Dopri5SolverExt(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
        **unused_kwargs
    ):
        #_handle_unused_kwargs(self, unused_kwargs)
        #del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)
        self.unused_kwargs = unused_kwargs

    def before_integrate(self, t):
        f0 = self.func(t[0].type_as(self.y0[0]), self.y0, **(self.unused_kwargs or {}))
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 4, self.rtol[0], self.atol[0], f0=f0, **self.unused_kwargs).to(t)
        else:
            first_step = _convert_to_tensor(0.01, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, interp_coeff=[self.y0] * 5)

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_dopri5_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _adaptive_dopri5_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_DORMAND_PRINCE_SHAMPINE_TABLEAU, **self.unused_kwargs)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        mean_sq_error_ratio = _compute_error_ratio(y1_error, atol=self.atol, rtol=self.rtol, y0=y0, y1=y1)
        accept_step = (torch.tensor(mean_sq_error_ratio) <= 1).all()

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        interp_coeff = _interp_fit_dopri5(y0, y1, k, dt) if accept_step else interp_coeff
        dt_next = _optimal_step_size(
            dt, mean_sq_error_ratio, safety=self.safety, ifactor=self.ifactor, dfactor=self.dfactor, order=5
        )
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state

    

