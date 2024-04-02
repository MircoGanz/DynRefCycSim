import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from CoolProp.CoolProp import PropsSI

class System:
    def __init__(self, evaporator, condenser, separator, compressor, expansion_valve, pi_controller):
        self.evaporator = evaporator
        self.condenser = condenser
        self.separator = separator
        self.compressor = compressor
        self.expansion_valve = expansion_valve
        self.pi_controller = pi_controller

    def odefun(self, t, x,
               m_evap_sec, θi_evap_sec,
               m_cond_sec, θi_cond_sec):
        

        h_evap, θw_evap, θl_evap, h_cond, θw_cond, θl_cond, p_sep, h_sep = np.split(x, [self.evaporator.N,
                                                                                        2 * self.evaporator.N,
                                                                                        3 * self.evaporator.N,
                                                                                        3 * self.evaporator.N + self.condenser.N,
                                                                                        3 * self.evaporator.N + 2 * self.condenser.N,
                                                                                        3 * self.evaporator.N + 3 * self.condenser.N,
                                                                                        3 * self.evaporator.N + 3 * self.condenser.N + 1])

        dxdt_evap = self.evaporator.odefun(np.concatenate([h_evap, θw_evap, θl_evap]), t, self.compressor.m, m_evap_sec, self.expansion_valve.ho, θi_evap_sec)
        
        dxdt_cond = self.condenser.odefun(np.concatenate([h_cond, θw_cond, θl_cond]), t, self.compressor.m, m_cond_sec, self.compressor.ho, θi_cond_sec)
        
        dxdt_sep = self.separator.odefun(np.array([p_sep, h_sep]), t, self.compressor.m, self.condenser.h[-1])

        dxdt = np.concatenate([dxdt_evap, dxdt_cond, dxdt_sep])

        return dxdt
                    
    def solve(self, t_span, po, m_evap_sec, p_evap_sec, θi_evap_sec, m_cond_sec, p_cond_sec, θi_cond_sec, refrigerant, liquid_evap, liquid_cond):

        x0 = np.concatenate([
            self.evaporator.h, self.evaporator.θw, self.evaporator.θl,
            self.condenser.h, self.condenser.θw, self.condenser.θl,
            [self.separator.p, self.separator.h]])

        self.evaporator.θr = PropsSI('T', 'P', po, 'H', self.evaporator.h, refrigerant)
        self.evaporator.ρr = PropsSI('D', 'P', po, 'H', self.evaporator.h, refrigerant)

        self.evaporator.cl = PropsSI('C', 'P', p_evap_sec, 'T', self.evaporator.θl, liquid_evap)
        self.evaporator.ρl = PropsSI('D', 'P', p_evap_sec, 'T', self.evaporator.θl, liquid_evap)

        self.condenser.θr = PropsSI('T', 'P', self.separator.p, 'H', self.condenser.h, refrigerant)
        self.condenser.ρr = PropsSI('D', 'P', self.separator.p, 'H', self.condenser.h, refrigerant)

        self.condenser.cl = PropsSI('C', 'P', p_cond_sec, 'T', self.condenser.θl, liquid_cond)
        self.condenser.ρl = PropsSI('D', 'P', p_cond_sec, 'T', self.condenser.θl, liquid_cond)

        self.separator.ρ = PropsSI('D', 'P', self.separator.p, 'H', self.separator.h, refrigerant)
        self.separator.dρdp = PropsSI('d(D)/d(P)|H', 'P', self.separator.p, 'H', self.separator.h, refrigerant)
        self.separator.dρdh = PropsSI('d(D)/d(H)|P', 'P', self.separator.p, 'H', self.separator.h, refrigerant)
        self.separator.hs = PropsSI('H', 'P', self.separator.p, 'Q', 0.0, refrigerant)
            
        sol = solve_ivp(self.odefun, t_span, x0,
                        args=(m_evap_sec, θi_evap_sec, m_cond_sec, θi_cond_sec),
                        vectorized=False, max_step=0.01, method='BDF')['y'][:,-1]

        h_evap, θw_evap, θl_evap, h_cond, θw_cond, θl_cond, p_sep, h_sep = np.split(sol,
                                                                                    [self.evaporator.N, 2 * self.evaporator.N,
                                                                                     3 * self.evaporator.N,
                                                                                     3 * self.evaporator.N + self.condenser.N,
                                                                                     3 * self.evaporator.N + 2 * self.condenser.N,
                                                                                     3 * self.evaporator.N + 3 * self.condenser.N,
                                                                                     3 * self.evaporator.N + 3 * self.condenser.N + 1])
        

        self.evaporator.h, self.evaporator.θw, self.evaporator.θl, self.evaporator.θr = h_evap, θw_evap, θl_evap, PropsSI(
            'T', 'H', h_evap, 'P', po, refrigerant)
        self.condenser.h, self.condenser.θw, self.condenser.θl, self.condenser.θr = h_cond, θw_cond, θl_cond, PropsSI(
            'T', 'H', h_cond, 'P', p_sep[0], refrigerant)
        self.separator.p, self.separator.h = p_sep[0], h_sep[0]


class HeatExchanger:

    def __init__(self, N, A, krw, klw, Cw, s, h_init, θw_init, θl_init):
        self.N = N
        self.A = A
        self.krw = krw
        self.klw = klw
        self.Cw = Cw
        self.s = s
        self.h = h_init
        self.θw = θw_init
        self.θl = θl_init
        self.dA = self.A / self.N
        self.dV = self.s * self.dA
        self.θr = np.zeros(self.N)
        self.ρr = np.zeros(self.N)

    def odefun(self, x, t, mr, ml, hrb, θlb):
        h, θw, θl = x.reshape((3, self.N))
        dxdt = np.zeros(3 * self.N)

        dxdt[0] = (mr * (hrb - h[0]) + self.krw * self.dA * (θw[0] - self.θr[0])) / (self.ρr[0] * self.dV)

        dxdt[1:self.N] = (mr * (h[:self.N - 1] - h[1:]) + self.krw * self.dA * (θw[1:] - self.θr[1:])) / (
                self.ρr[1:] * self.dV)

        dxdt[self.N:2 * self.N] = (self.krw * self.dA * (self.θr - θw) + self.klw * self.dA * (θl - θw)) / (
                self.Cw * self.dV)

        dxdt[2 * self.N:3 * self.N - 1] = (ml * self.cl[:-1] * (-θl[:-1] + θl[1:]) + self.klw * self.dA * (
                θw[:-1] - θl[:-1])) / (self.cl[:-1] * self.ρl[:-1] * self.dV)
        dxdt[-1] = (ml * self.cl[-1] * (-θl[-1] + θlb) + self.klw * self.dA * (θw[-1] - θl[-1])) / (
                self.cl[-1] * self.ρl[-1] * self.dV)

        return dxdt
    

class Separator:
    def __init__(self, V, p_init, h_init, refrigerant):
        self.V = V
        self.p = p_init
        self.h = h_init
        self.ho = PropsSI('H', 'P', self.p, 'Q', 0.0, refrigerant)

    def odefun(self, x, t, mr, hr):
        p, h = x[0], x[1]
        if h >= self.hs:
            ho = self.hs
        else:
            ho = h
        dpdt = -self.dρdh / self.dρdp * ((mr * (hr - ho)) / (self.V * (self.ρ + self.dρdh / self.dρdp)))
        dhdt = mr * (hr - ho) / (self.V * (self.ρ + self.dρdh / self.dρdp))
        self.ho = ho
        return np.append(dpdt, dhdt)


class Compressor:
    def __init__(self, Vd, eta_v, eta_s):
        self.Vd = Vd
        self.eta_v = eta_v
        self.eta_s = eta_s

    def model(self, f, hi, pi, po, refrigerant):
        s = PropsSI('S', 'P', pi, 'H', hi, refrigerant)
        hs = PropsSI('H', 'P', po, 'S', s, refrigerant)
        ρ = PropsSI('D', 'P', pi, 'H', hi, refrigerant)
        self.ho = hi + (hs - hi) / self.eta_s
        self.m = self.eta_v * f * self.Vd * ρ


class EV:
    def __init__(self, Kv):
        self.Kv = Kv

    def model(self, U, hi, pi, po, refrigerant):
        ρ = PropsSI('D', 'P', pi, 'H', hi, refrigerant)
        self.m = self.Kv * U * np.sqrt((pi - po) * 1e-5 * 1000 * ρ) / 3600
        self.ho = hi


class PIController:

    def __init__(self, y_init, Kp, Ki):
        self.y = y_init
        self.Kp = Kp
        self.Ki = Ki

    def model(self, dt, e):
        self.y -= self.Ki * e * dt
        return self.y - self.Kp * e