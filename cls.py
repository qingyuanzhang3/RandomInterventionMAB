import numpy as np
from math import exp
from scipy.integrate import quad, dblquad
from scipy.special import pbdv, gamma
from scipy.interpolate import PchipInterpolator, CubicSpline

log_eps = -np.log(np.finfo(float).eps)
log_max = np.log(np.finfo(float).max)


def _poisson_arrival_times(T, lbda):
    batch = max(1, int(2 * T * lbda))
    times = np.cumsum(np.random.exponential(1 / lbda, batch))
    while times[-1] < T:
        times = np.hstack((times, times[-1] + np.cumsum(np.random.exponential(1/lbda, batch))))
    return np.hstack((0.0, times[times < T]))


def _apply_f(f, arr):
    arr = np.asarray(arr, dtype=float)
    try:
        out = np.asarray(f(arr), dtype=float)
        if out.shape == arr.shape:
            return out
    except (TypeError, ValueError):
        pass
    return np.array([f(v) for v in arr], dtype=float)


def _exp_kernel_integral(xs, f, alpha):
    xs = np.asarray(xs, dtype=float)
    return np.array([quad(lambda y: f(x + y) * np.exp(-alpha * y), 0, np.inf)[0]
                     for x in xs])


class BM:
    def __init__(self, x, sigma, f=None, lbda=None, q=None, T=None):
        '''Brownian motion.

        Simulates a path of X observed at the Poissonian intervention times up to
        the horizon T (self.Tscr) and evaluates the Gittins index at
        those observation points (self.GittinsIndTscr).

        Parameters
        ----------
        x : float
            Initial value of the process.
        sigma : float
            Brownian motion coefficient.
        q : float
            Discount rate.
        lbda : float
            Intervention rate.
        f : callable
            Reward function.
            '''
        self.x0 = x
        self.sigma = sigma
        self.q = q
        self.lbda = lbda
        self.qlbda = self.q + self.lbda
        self.roots_q = np.sqrt(self.q * 2) / self.sigma
        self.roots_qlbda = np.sqrt(self.qlbda * 2) / self.sigma
        self.f = f

        self.Tscr = _poisson_arrival_times(T, self.lbda)
        self.XTscr = self.Xt(self.Tscr)

        J = _exp_kernel_integral(self.XTscr, self.f, self.roots_q)
        self.fx = _apply_f(self.f, self.XTscr)
        self.GittinsIndTscr = (self.roots_q / self.roots_qlbda) * (
            self.fx + (self.roots_qlbda - self.roots_q) * J
        )
        self.ctsGittinsIndTscr = self.roots_q * J

    def Xt(self, ts):
        dt = np.diff(ts)
        path = np.cumsum(np.random.randn(dt.size) * np.sqrt(dt))
        return np.hstack((0, path)) * self.sigma + self.x0


class OU:
    def __init__(self, x, gam, f=None, lbda=None, q=None, T=None, interp=None):
        '''One parameter Ornstein-Uhlenbeck process.

        Simulates a path of X observed at the Poissonian intervention times up to
        the horizon T (self.Tscr) and evaluates the Gittins index at
        those observation points (self.GittinsIndTscr).

        Parameters
        ----------
        x : float
            Initial value of the process.
        gam : float
            Drift.
        q : float
            Discount rate.
        lbda : float
            Intervention rate.
        f : callable
            Reward function.
        interp : scipy interpolator, optional
            Precomputed interpolator of the Gittins index, as returned by
            self.build_gittins_interp. When given, the index at the observation points is
            evaluated by interpolation (self.GittinsInd_fast) instead of by the
            double integration in self.GittinsInd. If None, the double integration is used.
            '''
        self.x0 = x
        self.gamma = gam
        self.q = q
        self.lbda = lbda
        self.qlbda = self.q + self.lbda
        self.Wq = 2 * np.sqrt(np.pi * self.gamma) / (gamma(self.q / self.gamma))
        self.Wqlbda = 2 * np.sqrt(np.pi * self.gamma) / (gamma(self.qlbda / self.gamma))
        self.sq2g = np.sqrt(2 * self.gamma)
        self.m = lambda y: 2 * np.exp(-self.gamma * y ** 2)
        self.f = f

        self.Tscr = _poisson_arrival_times(T, self.lbda)
        self.XTscr = self.Xt(self.Tscr)
        self.fx = _apply_f(self.f, self.XTscr)
        if interp:
            self._g_interp = interp
            self.GittinsIndTscr = np.array([self.GittinsInd_fast(x) for x in self.XTscr])
        else:
            self.GittinsIndTscr = np.array([self.GittinsInd(x) for x in self.XTscr])
            

    def phi_q(self, x):
        return np.exp(self.gamma * x ** 2 / 2) * pbdv(-self.q / self.gamma, x * self.sq2g)[0]
    
    def phi_qlbda(self, x):
        return np.exp(self.gamma * x ** 2 / 2) * pbdv(-self.qlbda / self.gamma, x * self.sq2g)[0]

    def phi_ratio(self, x, y):
        return self.phi_q(y)/self.phi_q(x)

    def Green_q(self, x, y):
        return 1 / self.Wq * self.phi_q(-x) * self.phi_q(y)
    
    def Green_qlbda(self, x, y):
        return 1 / self.Wqlbda * self.phi_qlbda(-x) * self.phi_qlbda(y)

    def Wt(self, ts):
        dt = np.diff(ts)
        path = np.cumsum(np.random.randn(dt.size) * np.sqrt(dt))
        return np.hstack((0, path)) + self.x0

    def Xt(self, ts):
        ts_transformed = (np.exp(2 * self.gamma * ts) - 1) / 2 / self.gamma
        Wt = self.Wt(ts_transformed)
        return np.exp(-self.gamma * ts) * Wt

    def H(self, x, h):
        if self.gamma * x ** 2 > log_max / 2:
            return h(x)
        upper = np.sqrt(x ** 2 + 2 * log_eps / self.gamma)
        t1 = h(x) * (1 - self.lbda * quad
                     (lambda y: self.phi_ratio(x, y) * self.m(y) * self.Green_qlbda(x, y), x, upper)[0])
        t2 = self.lbda * quad(lambda y: h(y) * self.m(y) * self.Green_qlbda(x, y), x, upper)[0]
        t3_1 = self.lbda**2 * dblquad(
            lambda u, z: self.Green_qlbda(x, z) * h(u)
            * (self.Green_q(z, u) - self.phi_ratio(x, z) * self.Green_q(x, u))
            * self.m(u) * self.m(z), x, upper, lambda z: z, lambda z: upper)[0]
        t3_2 = self.lbda**2 * dblquad(
            lambda u, z: self.Green_qlbda(x, z) * h(u)
            * (self.Green_q(u, z) - self.phi_ratio(x, z) * self.Green_q(x, u))
            * self.m(u) * self.m(z), x, upper, x, lambda z: z)[0]
        return t1 + t2 + t3_1 + t3_2

    def GittinsInd(self, x):
        return self.H(x, self.f) / self.H(x, lambda y: 1)
    
    def build_gittins_interp(self, n=500, kind="pchip", num_sd=5.):
        bound = num_sd / np.sqrt(2 * self.gamma)
        xs = np.linspace(-bound, bound, n)
        ys = np.fromiter((self.GittinsInd(x) for x in xs), float, count=n)

        Interp = {"pchip": PchipInterpolator, "cubic": CubicSpline}[kind]
        self._g_interp = Interp(xs, ys, extrapolate=True)
        self._g_kind = kind
        return self._g_interp
    
    def save_gittins_interp(self, filename):
        xs = self._g_interp.x
        ys = self._g_interp(xs)
        np.savez(filename, xs=xs, ys=ys, kind=self._g_kind)

    def GittinsInd_fast(self, x):
        xa = np.atleast_1d(np.asarray(x, float))
        y = self._g_interp(xa, extrapolate=True)
        return y.reshape(np.shape(x)) if np.ndim(x) else float(y[0])


class RSNLP_rational:
    def __init__(self, x, mu, r, l, sigma, b, f=None, lbda=None, q=None, T=None):
        '''Reflected spectrally negative Levy process with exponential jumps.

        Simulates a path of X observed at the Poissonian intervention times up to
        the horizon T (self.Tscr) and evaluates the Gittins index at
        those observation points (self.GittinsIndTscr).

        Parameters
        ----------
        x : float
            Initial value of the process.
        mu : float
            Drift.
        r : float
            Rate of the exponential jump distribution; mean jump size 1/r.
        sigma : float
            Brownian motion coefficient.
        l : float
            Arrival rate of the compound Poisson jumps.
        q : float
            Discount rate.
        lbda : float
            Intervention rate.
        f : callable
            Reward function.
        b : float
            Reflection level.
            '''
        self.x0 = x
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.q = q
        self.l = l
        self.lbda = lbda
        self.qlbda = self.q + self.lbda
        self.b = b
        self.roots_q = np.sort(np.roots([1 / 2 * self.sigma ** 2, (self.mu + 1 / 2 * self.sigma ** 2 * self.r),
                                         (self.mu * self.r - self.l - self.q), -self.q * self.r]))
        self.roots_qlbda = np.sort(np.roots([1 / 2 * self.sigma ** 2, (self.mu + 1 / 2 * self.sigma ** 2 * self.r),
                                             (self.mu * self.r - self.l - (self.q + self.lbda)), -(self.q + self.lbda) * self.r]))
        self.f = f

        self.psi_prime = lambda s: self.mu + self.sigma ** 2 * s + self.l * s / (s + self.r) ** 2 - self.l / (s + self.r)
        self.Z_qlbda = lambda x: (x > 0) * self.qlbda * ((np.exp(self.roots_qlbda[-1] * x) - 1) / (self.roots_qlbda[-1] * self.psi_prime(self.roots_qlbda[-1]))
                    + (np.exp(self.roots_qlbda[0] * x) - 1) / (self.roots_qlbda[0] * self.psi_prime(self.roots_qlbda[0]))
                    + (np.exp(self.roots_qlbda[1] * x) - 1) / (self.roots_qlbda[1] * self.psi_prime(self.roots_qlbda[1]))) + 1

        phi_q = self.roots_q[-1]
        self.integral1_barrier = self.lbda / self.q * phi_q * quad(
            lambda y: self.f(self.b + y) * np.exp(-phi_q * y), 0, np.inf)[0]
        self.integral2_barrier = self.lbda / self.q

        self.Tscr = _poisson_arrival_times(T, self.lbda)
        self.ts = np.sort(np.hstack((np.linspace(0, T, 1000), self.Tscr[1:])))
        self.Xts = self.Xt(self.ts)
        self.mask = np.isin(self.ts, self.Tscr)
        self.XTscr = self.Xts[self.mask]

        J = _exp_kernel_integral(self.XTscr, self.f, phi_q)
        self.fx = _apply_f(self.f, self.XTscr)
        self.ctsGittinsIndTscr = phi_q * J

        idx = np.empty_like(self.XTscr, dtype=float)
        barrier_val = (self.f(self.b) + self.integral1_barrier) / (1 + self.integral2_barrier)
        above = self.XTscr > self.b
        a = -self.b + self.XTscr[above]
        c = self.lbda / self.qlbda * self.Z_qlbda(a) / self.Z_qlbda_phiq(a)
        idx[above] = self.fx[above] * (1 - c) + c * phi_q * J[above]
        idx[~above] = barrier_val
        self.GittinsIndTscr = idx

    def Z_qlbda_phiq(self, x):
        p1 = (np.exp(self.roots_qlbda[-1] * x) - np.exp(self.roots_q[-1] * x)) / ((self.roots_qlbda[-1] - self.roots_q[-1]) * self.psi_prime(self.roots_qlbda[-1]))
        p2 = (np.exp(self.roots_qlbda[0] * x) - np.exp(self.roots_q[-1] * x)) / ((self.roots_qlbda[0] - self.roots_q[-1]) * self.psi_prime(self.roots_qlbda[0]))
        p3 = (np.exp(self.roots_qlbda[1] * x) - np.exp(self.roots_q[-1] * x)) / ((self.roots_qlbda[1] - self.roots_q[-1]) * self.psi_prime(self.roots_qlbda[1]))
        return (x > 0) * (np.exp(self.roots_q[-1] * x) + self.lbda * (p1 + p2 + p3)) + np.exp(self.roots_q[-1] * x) * (x <= 0)

    def Xt(self, ts):
        dt = np.diff(ts)
        Nt = np.random.poisson(self.l * dt)
        jumps = np.random.gamma(Nt, 1.0 / self.r)
        incr = self.mu * dt - jumps + self.sigma * np.sqrt(dt) * np.random.randn(dt.size)
        X = np.concatenate(([self.x0], self.x0 + np.cumsum(incr)))
        return X - np.minimum.accumulate(np.minimum(0, X - self.b))


class RBM:
    def __init__(self, x, b, sigma, f=None, lbda=None, q=None, T=None):
        '''Reflected Brownian motion.

        Simulates a path of X observed at the Poissonian intervention times up to
        the horizon T (self.Tscr) and evaluates the Gittins index at
        those observation points (self.GittinsIndTscr).

        Parameters
        ----------
        x : float
            Initial value of the process; must satisfy x > b.
        sigma : float
            Brownian motion coefficient.
        b : float
            Reflection level.
        q : float
            Discount rate.
        lbda : float
            Intervention rate.
        f : callable
            Reward function.
        '''
        self.x0 = x
        self.sigma = sigma
        self.q = q
        self.lbda = lbda
        self.qlbda = self.q + self.lbda
        self.b = b
        self.roots_q = np.sqrt(self.q * 2) / self.sigma
        self.roots_qlbda = np.sqrt(self.qlbda * 2) / self.sigma
        self.f = f
        self.Z_qlbda = lambda x: (x > 0) *  np.cosh(x / self.sigma * np.sqrt(2 * self.qlbda)) + 1 * (x <= 0)

        self.Tscr = _poisson_arrival_times(T, self.lbda)
        self.XTscr = self.Xt(self.Tscr)

        J = _exp_kernel_integral(self.XTscr, self.f, self.roots_q)
        self.fx = _apply_f(self.f, self.XTscr)
        self.ctsGittinsIndTscr = self.roots_q * J

        a = -self.b + self.XTscr
        c = self.lbda / self.qlbda * self.Z_qlbda(a) / self.Z_qlbda_phiq(a)
        self.GittinsIndTscr = self.fx * (1 - c) + c * self.roots_q * J
        self.ctsGittinsInd = lambda x: _exp_kernel_integral([x], self.f, self.roots_q) * self.roots_q

    def Z_qlbda_phiq(self, x):
        positive = np.cosh(x / self.sigma * np.sqrt(2 * self.qlbda)) + np.sqrt(self.q/self.qlbda) * np.sinh(x / self.sigma * np.sqrt(2 * self.qlbda))
        return (x > 0) * positive + np.exp(self.roots_q * x) * (x <= 0)

    def Xt(self, ts):
        dt = np.diff(ts)
        incr = self.sigma * np.sqrt(dt) * np.random.randn(dt.size)
        X = np.concatenate(([0], np.cumsum(incr)))
        return self.b + np.abs(self.x0 - self.b + X)


class SNLP_rational:
    def __init__(self, x, mu, r, l, sigma, f=None, lbda=None, q=None, T=None):
        '''Spectrally negative Levy process with exponential jumps.

        Simulates a path of X observed at the Poissonian intervention times up to
        the horizon T (self.Tscr) and evaluates the Gittins index at
        those observation points (self.GittinsIndTscr).

        Parameters
        ----------
        x : float
            Initial value of the process.
        mu : float
            Drift.
        r : float
            Rate of the exponential jump distribution; mean jump size 1/r.
        sigma : float
            Brownian motion coefficient.
        l : float
            Arrival rate of the compound Poisson jumps.
        q : float
            Discount rate.
        lbda : float
            Intervention rate.
        f : callable
            Reward function.
        '''
        self.x0 = x
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.q = q
        self.l = l
        self.lbda = lbda
        self.roots_q = np.sort(np.roots([1 / 2 * self.sigma ** 2, (self.mu + 1 / 2 * self.sigma ** 2 * self.r),
                                        (self.mu * self.r - self.l - self.q), -self.q * self.r]))
        self.roots_qlbda = np.sort(np.roots([1 / 2 * self.sigma ** 2, (self.mu + 1 / 2 * self.sigma ** 2 * self.r),
                                            (self.mu * self.r - self.l - (self.q + self.lbda)), -(self.q + self.lbda) * self.r]))
        self.f = f

        self.Tscr = _poisson_arrival_times(T, self.lbda)
        self.XTscr = self.Xt(self.Tscr)

        phi_q, phi_qlbda = self.roots_q[-1], self.roots_qlbda[-1]
        J = _exp_kernel_integral(self.XTscr, self.f, phi_q)
        self.fx = _apply_f(self.f, self.XTscr)
        self.GittinsIndTscr = (phi_q / phi_qlbda) * (self.fx + (phi_qlbda - phi_q) * J)
        self.ctsGittinsIndTscr = phi_q * J

    def Xt(self, ts):
        dt = np.diff(ts)
        Nt = np.random.poisson(self.l * dt)
        jumps = np.random.gamma(Nt, 1.0 / self.r)
        incr = self.mu * dt - jumps + self.sigma * np.sqrt(dt) * np.random.randn(dt.size)
        return np.concatenate(([self.x0], self.x0 + np.cumsum(incr)))


# Simulation
def _precompute(Xs, q):
    for X in Xs:
        if getattr(X, "fx", None) is None:
            X.fx = _apply_f(X.f, X.XTscr)
        X.dTs = np.diff(X.Tscr)
        X.period_factor = (1.0 - np.exp(-q * X.dTs)) / q


def _run_strategy(Xs, T, q, indices):
    n = len(Xs)
    for X in Xs:
        X.i = 0
    cur = [indices[k][0] for k in range(n)]
    Tn = 0.0
    reward = 0.0
    while Tn < T:
        arm = max(range(n), key=cur.__getitem__)
        X = Xs[arm]
        i = X.i
        fval = X.fx[i]
        discount = exp(-Tn * q)
        if i + 1 < len(X.Tscr):
            dT = X.dTs[i]
            if Tn + dT > T:
                reward += discount * fval * (1.0 - exp(-q * (T - Tn))) / q
                break
            reward += discount * fval * X.period_factor[i]
            Tn += dT
            X.i = i + 1
            cur[arm] = indices[arm][i + 1]
        else:
            reward += discount * fval * (1.0 - exp(-q * (T - Tn))) / q
            break
    return reward


def ComputePathwise(Xs, T, q, cts_gittins=False, myopic=False):
    _precompute(Xs, q)
    out = [_run_strategy(Xs, T, q, [X.GittinsIndTscr for X in Xs])]
    if myopic:
        out.append(_run_strategy(Xs, T, q, [X.fx for X in Xs]))
    if cts_gittins:
        out.append(_run_strategy(Xs, T, q, [X.ctsGittinsIndTscr for X in Xs]))
    return tuple(out)
    