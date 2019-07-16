# Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
import torch
import torch.nn as nn

import collections

from torchdiffeq._impl.misc import (
    _scaled_dot_product, _convert_to_tensor, _is_finite, _is_iterable,
    _optimal_step_size, _compute_error_ratio
)
from torchdiffeq._impl.solvers import AdaptiveStepsizeODESolver
from torchdiffeq._impl.interp import _interp_fit, _interp_evaluate
from torchdiffeq._impl.rk_common import _RungeKuttaState, _ButcherTableau
from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.misc import _flatten, _flatten_convert_none_to_zeros, _decreasing, _norm
from torchdiffeq._impl.adams import (
    _VCABMState, g_and_explicit_phi, compute_implicit_phi, _MAX_ORDER, _MIN_ORDER, gamma_star
)

from odeint_ext.odeint_ext_misc import _select_initial_step


class VariableCoefficientAdamsBashforthExt(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, implicit=True, max_order=_MAX_ORDER, safety=0.9, ifactor=10.0, dfactor=0.2,
        **unused_kwargs
    ):
        #_handle_unused_kwargs(self, unused_kwargs)
        #del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.implicit = implicit
        self.max_order = int(max(_MIN_ORDER, min(max_order, _MAX_ORDER)))
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.unused_kwargs = unused_kwargs

    def before_integrate(self, t):
        prev_f = collections.deque(maxlen=self.max_order + 1)
        prev_t = collections.deque(maxlen=self.max_order + 1)
        phi = collections.deque(maxlen=self.max_order)

        t0 = t[0]
        f0 = self.func(t0.type_as(self.y0[0]), self.y0, **self.unused_kwargs)
        prev_t.appendleft(t0)
        prev_f.appendleft(f0)
        phi.appendleft(f0)
        first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0, **self.unused_kwargs).to(t)

        self.vcabm_state = _VCABMState(self.y0, prev_f, prev_t, next_t=t[0] + first_step, phi=phi, order=1)

    def advance(self, final_t):
        final_t = _convert_to_tensor(final_t).to(self.vcabm_state.prev_t[0])
        while final_t > self.vcabm_state.prev_t[0]:
            self.vcabm_state = self._adaptive_adams_step(self.vcabm_state, final_t)
        assert final_t == self.vcabm_state.prev_t[0]
        return self.vcabm_state.y_n

    def _adaptive_adams_step(self, vcabm_state, final_t):
        y0, prev_f, prev_t, next_t, prev_phi, order = vcabm_state
        if next_t > final_t:
            next_t = final_t
        dt = (next_t - prev_t[0])
        dt_cast = dt.to(y0[0])

        # Explicit predictor step.
        g, phi = g_and_explicit_phi(prev_t, next_t, prev_phi, order)
        g = g.to(y0[0])
        p_next = tuple(
            y0_ + _scaled_dot_product(dt_cast, g[:max(1, order - 1)], phi_[:max(1, order - 1)])
            for y0_, phi_ in zip(y0, tuple(zip(*phi)))
        )

        # Update phi to implicit.
        next_f0 = self.func(next_t.to(p_next[0]), p_next, **self.unused_kwargs)
        implicit_phi_p = compute_implicit_phi(phi, next_f0, order + 1)

        # Implicit corrector step.
        y_next = tuple(
            p_next_ + dt_cast * g[order - 1] * iphi_ for p_next_, iphi_ in zip(p_next, implicit_phi_p[order - 1])
        )

        # Error estimation.
        tolerance = tuple(
            atol_ + rtol_ * torch.max(torch.abs(y0_), torch.abs(y1_))
            for atol_, rtol_, y0_, y1_ in zip(self.atol, self.rtol, y0, y_next)
        )
        local_error = tuple(dt_cast * (g[order] - g[order - 1]) * iphi_ for iphi_ in implicit_phi_p[order])
        error_k = _compute_error_ratio(local_error, tolerance)
        accept_step = (torch.tensor(error_k) <= 1).all()

        if not accept_step:
            # Retry with adjusted step size if step is rejected.
            dt_next = _optimal_step_size(dt, error_k, self.safety, self.ifactor, self.dfactor, order=order)
            return _VCABMState(y0, prev_f, prev_t, prev_t[0] + dt_next, prev_phi, order=order)

        # We accept the step. Evaluate f and update phi.
        next_f0 = self.func(next_t.to(p_next[0]), y_next, **self.unused_kwargs)
        implicit_phi = compute_implicit_phi(phi, next_f0, order + 2)

        next_order = order

        if len(prev_t) <= 4 or order < 3:
            next_order = min(order + 1, 3, self.max_order)
        else:
            error_km1 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 1] - g[order - 2]) * iphi_ for iphi_ in implicit_phi_p[order - 1]), tolerance
            )
            error_km2 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 2] - g[order - 3]) * iphi_ for iphi_ in implicit_phi_p[order - 2]), tolerance
            )
            if min(error_km1 + error_km2) < max(error_k):
                next_order = order - 1
            elif order < self.max_order:
                error_kp1 = _compute_error_ratio(
                    tuple(dt_cast * gamma_star[order] * iphi_ for iphi_ in implicit_phi_p[order]), tolerance
                )
                if max(error_kp1) < max(error_k):
                    next_order = order + 1

        # Keep step size constant if increasing order. Else use adaptive step size.
        dt_next = dt if next_order > order else _optimal_step_size(
            dt, error_k, self.safety, self.ifactor, self.dfactor, order=order + 1
        )

        prev_f.appendleft(next_f0)
        prev_t.appendleft(next_t)
        return _VCABMState(p_next, prev_f, prev_t, next_t + dt_next, implicit_phi, order=next_order)



