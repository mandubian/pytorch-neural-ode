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


from odeint_ext.odeint_ext_dopri5 import Dopri5SolverExt
SOLVERS["dopri5-ext"] = Dopri5SolverExt

from odeint_ext.odeint_ext_adams import VariableCoefficientAdamsBashforthExt
SOLVERS["adams-ext"] = VariableCoefficientAdamsBashforthExt


def _check_inputs_custom(func, y0, t, **kwargs):
    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)
        _base_nontuple_func_ = func
        def replace_func(t, y, **kwargs):
            return (_base_nontuple_func_(t, y[0], **kwargs), )
        #func = lambda t, y: (_base_nontuple_func_(t, y[0]),)
        func = replace_func
    assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(type(y0_))

    if _decreasing(t):
        t = -t
        _base_reverse_func = func
        def replace_func(t, y, **kwargs):
            return tuple(-f_ for f_ in _base_reverse_func(-t, y, **(kwargs or {})))
        #func = lambda t, y: tuple(-f_ for f_ in _base_reverse_func(-t, y))
        func = replace_func

    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.type()))
    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, func, y0, t, kwargs



def odeint_ext(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a Tensor holding the state `y` and a scalar Tensor
            `t` into a Tensor of state derivatives with respect to time.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. May
            have any floating point or complex dtype.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. May have any floating
            point dtype. Converted to a Tensor with float64 dtype.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicatcing the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        name: Optional name for this operation.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
        TypeError: if `options` is supplied without `method`, or if `t` or `y0` has
            an invalid dtype.
    """

    tensor_input, func, y0, t, options = _check_inputs_custom(func, y0, t, **(options or {}))

    if options is None:
        options = {}
    elif method is None:
        raise ValueError('cannot supply `options` without specifying `method`')

    if method is None:
        method = 'dopri5'

    solver = SOLVERS[method](func, y0, rtol=rtol, atol=atol, **(options or {}))
    solution = solver.integrate(t)

    if tensor_input:
        solution = solution[0]
    return solution




class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 8, 'Internal error: all arguments required.'
        y0, func, t, flat_params, rtol, atol, method, options = \
            args[:-7], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]

        ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options = func, rtol, atol, method, options

        with torch.no_grad():
            ans = odeint_ext(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, flat_params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        t, flat_params, *ans = ctx.saved_tensors
        ans = tuple(ans)
        func, rtol, atol, method, options = ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options
        n_tensors = len(ans)
        f_params = tuple(func.parameters())

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug, **kwargs):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                # ADDED FOR EXT
                # as options can also contain grad tensors, we need to manage those too
                grad_options = {}
                for k, option in options.items():
                    if option is not None and option.requires_grad:
                        option = option.detach().requires_grad_(True)
                    grad_options[k] = option
                func_eval = func(t, y, **grad_options)
                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )
            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*func_eval, *vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = func(t[i], ans_i, **options)
                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = 0
                for func_i_, grad_output_i_ in zip(func_i, grad_output_i):                    
                    # if you have more than one gradient in input
                    #if grad_output_i_.dim() > func_i_.dim():
                    #    grad_output_i_ = grad_output_i_.reshape(grad_output_i_.shape[0], -1)
                    #    func_i_ = func_i_.reshape(-1)
                    #    s = torch.matmul(grad_output_i_, func_i_).sum().unsqueeze(0)
                    #    dLd_cur_t += s                
                    #else:
                        
                    dLd_cur_t += sum(
                        torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    ).unsqueeze(0)
                
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)
                aug_ans = odeint_ext(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]), rtol=rtol, atol=atol, method=method, options=options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[n_tensors:2 * n_tensors]
                adj_time = aug_ans[2 * n_tensors]
                adj_params = aug_ans[2 * n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            return (*adj_y, None, time_vjps, adj_params, None, None, None, None, None)


def odeint_adjoint_ext(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    if torch.is_tensor(y0):

        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y, **options):
                return (self.base_func(t, y[0], **options),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    flat_params = _flatten(func.parameters())
    ys = OdeintAdjointMethod.apply(*y0, func, t, flat_params, rtol, atol, method, options)

    if tensor_input:
        ys = ys[0]
    return ys





class OdeintDTAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 10, 'Internal error: all arguments required.'
        y0, func, t_begin, t_end, dt, flat_params, rtol, atol, method, options = \
            args[:-9], args[-9], args[-8], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]

        ctx.func, ctx.t_begin, ctx.t_end, ctx.rtol, ctx.atol, ctx.method, ctx.options = func, t_begin, t_end, rtol, atol, method, options

        n = int(abs(t_end - t_begin) / dt)
        ts = torch.linspace(t_begin, t_end, n).to(y0[0])
        with torch.no_grad():
            ans = odeint_ext(func, y0, ts, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(dt, flat_params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        dt, flat_params, *ans = ctx.saved_tensors
        ans = tuple(ans)
        func, t_begin, t_end, rtol, atol, method, options = ctx.func, ctx.t_begin, ctx.t_end, ctx.rtol, ctx.atol, ctx.method, ctx.options
        n_tensors = len(ans)
        f_params = tuple(func.parameters())
        
        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug, **kwargs):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                # ADDED FOR EXT
                # as options can also contain grad tensors, we need to manage those too
                grad_options = {}
                if options is not None:
                    for k, option in options.items():
                        if option.requires_grad:
                            option = option.detach().requires_grad_(True)
                        grad_options[k] = option
                func_eval = func(t, y, **grad_options)
                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )
            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*func_eval, *vjp_y, vjp_t, vjp_params)

        #T = ans[0].shape[0]
        # compute time depth using boundaries and dt
        T = int(abs(t_end - t_begin)/dt)
        
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(dt)
            time_vjps = []
            for i in range(T - 1, 0, -1):
                t_i = t_begin + i * dt
                t_i_1 = t_begin + (i-1) * dt
                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                #func_i = func(t[i], ans_i, **options)
                func_i = func(t_i, ans_i, **(options or {}))
                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = 0
                for func_i_, grad_output_i_ in zip(func_i, grad_output_i):                    
                    dLd_cur_t += sum(
                        torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    ).unsqueeze(0)
                
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)
                aug_ans = odeint_ext(
                    augmented_dynamics, aug_y0,
                    #torch.tensor([t[i], t[i - 1]]), rtol=rtol, atol=atol, method=method, options=options
                    torch.tensor([t_i, t_i_1]), rtol=rtol, atol=atol, method=method, options=options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[n_tensors:2 * n_tensors]
                adj_time = aug_ans[2 * n_tensors]
                adj_params = aug_ans[2 * n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])
            #adj_time = time_vjps[0] / time_vjps.shape[0]
            adj_time = time_vjps.mean()
            
                   #y0    , func, t_begin, t_end, dt,       flat_params, rtol, atol, method, option
            return (*adj_y, None, None,    None,  adj_time, adj_params,  None, None, None,   None, None)


def odeint_adjoint_dt_ext(func, y0, t_begin, t_end, dt, rtol=1e-6, atol=1e-12, method=None, options=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    if torch.is_tensor(y0):

        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y, **options):
                return (self.base_func(t, y[0], **options),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    flat_params = _flatten(func.parameters())
    ys = OdeintDTAdjointMethod.apply(*y0, func, t_begin, t_end, dt, flat_params, rtol, atol, method, options)

    if tensor_input:
        ys = ys[0]
    return ys


