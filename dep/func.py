from dep.func import *
from dep.classes import *
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from CoolProp.CoolProp import PropsSI
from ipywidgets import interact, FloatSlider, widgets, Button
import threading
from IPython.display import display, clear_output
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plotlogph(ax, refrigerant):
    T_min = PropsSI('Tmin', refrigerant) + 50
    T_max = PropsSI('Tcrit', refrigerant)
    temperatures = np.linspace(T_min, T_max, 300)

    p_sat = []
    h_liq = []
    h_vap = []

    for T in temperatures:
        p = PropsSI('P', 'T', T, 'Q', 0, refrigerant)
        h_l = PropsSI('H', 'T', T, 'Q', 0, refrigerant)
        h_v = PropsSI('H', 'T', T, 'Q', 1, refrigerant)

        p_sat.append(p)
        h_liq.append(h_l)
        h_vap.append(h_v)

    p_sat = np.array(p_sat) / 1e5

    ax.semilogy(h_liq, p_sat, 'r')
    ax.semilogy(h_vap, p_sat, 'r')
    ax.set_xlabel('specific Enthalpy / J/kg')
    ax.set_ylabel('Pressure / bar')
    ax.set_xlim([200000, 500000])
    ax.grid(True)
    
def adjust_slider_value(slider, actual_value, max_step):
    target_value = slider.value
    difference = target_value - actual_value
    if abs(difference) > max_step:
        actual_value += np.sign(difference) * max_step
    else:
        actual_value = target_value
    return actual_value
    

def simulation_loop(f_slider, m_evap_sec_slider, θi_evap_sec_slider, m_cond_sec_slider, θi_cond_sec_slider, SHsp_slider, control, fig):
     
    p_evap_sec = 1e5
    liquid_evap = 'INCOMP::TY20'

    p_cond_sec = 1e5
    liquid_cond = 'Water'

    refrigerant = 'R134a'

    V, p_init_s, h_init_s = 0.1, 994666.5285049919, 269880.1766220895
    separator = Separator(V, p_init_s, h_init_s, refrigerant)

    Nv = 50
    h_init_v = np.array([258033.80094254, 260885.49673303, 263753.07250677, 266636.60458444,
                           269536.17077165, 272451.8502271 , 275383.72335134, 278331.87169232,
                           281296.37786486, 284277.32548168, 287274.79909416, 290288.88414121,
                           293319.666905  , 296367.23447259, 299431.67470249, 302513.07619552,
                           305611.52826935, 308727.12093618, 311859.94488319, 315010.09145538,
                           318177.65264053, 321362.72105591, 324565.38993667, 327785.75312564,
                           331023.90506432, 334279.94078507, 337553.95590423, 340846.04661614,
                           344156.30968791, 347484.842455  , 350831.74281732, 354197.10923618,
                           357581.04073315, 360983.63688486, 364404.9978218 , 367845.22422714,
                           371304.41733581, 374782.67893389, 378280.11135819, 381796.81749645,
                           385332.90078779, 388888.46522407, 392463.61535275, 395717.42621795,
                           398104.61685418, 399857.12408564, 401144.17339858, 402089.84440878,
                           402785.06322642, 403296.45815245])

    θw_init_v = np.array([272.60487027, 272.63763657, 272.67058411, 272.70371377,
                           272.73702645, 272.77052305, 272.80420446, 272.83807159,
                           272.87212535, 272.90636666, 272.94079643, 272.97541558,
                           273.01022505, 273.04522575, 273.08041863, 273.11580463,
                           273.15138468, 273.18715974, 273.22313076, 273.2592987 ,
                           273.29566451, 273.33222917, 273.36899364, 273.40595891,
                           273.44312595, 273.48049574, 273.51806929, 273.55584757,
                           273.59383161, 273.63202239, 273.67042093, 273.70902825,
                           273.74784537, 273.78687332, 273.82611313, 273.86556583,
                           273.90523247, 273.94511409, 273.98521176, 274.02552653,
                           274.06605948, 274.10681167, 274.1477842 , 274.56451127,
                           275.50556826, 276.20355052, 276.71876009, 277.09898214,
                           277.37967728, 277.58701485])


    θl_init_v = np.array([275.58581315, 275.63503564, 275.68452637, 275.73428664,
                           275.78431777, 275.83462108, 275.8851979 , 275.93604958,
                           275.98717744, 276.03858285, 276.09026716, 276.14223173,
                           276.19447794, 276.24700717, 276.2998208 , 276.35292023,
                           276.40630687, 276.45998212, 276.51394741, 276.56820416,
                           276.62275381, 276.6775978 , 276.73273758, 276.78817463,
                           276.8439104 , 276.89994638, 276.95628405, 277.01292492,
                           277.06987049, 277.12712227, 277.1846818 , 277.2425506 ,
                           277.30073023, 277.35922223, 277.41802818, 277.47714965,
                           277.53658822, 277.59634549, 277.65642306, 277.71682256,
                           277.77754561, 277.83859385, 277.89996893, 277.96167251,
                           278.01753485, 278.05884963, 278.08936792, 278.11191463,
                           278.12857683, 278.14089371])

    Av, krw_v, klw_v, Cw_v, s_v = 2.5, 500, 1000, 3532500, 0.004
    evaporator = HeatExchanger(Nv, Av, krw_v, klw_v, Cw_v, s_v, h_init_v, θw_init_v, θl_init_v)

    Nc = 50
    h_init_c = np.array([438189.58466526, 430507.30822852, 425083.79353461, 421231.92242608,
                           418231.85932632, 415209.77912001, 412165.5583012 , 409099.07496389,
                           406010.20909672, 402898.84280942, 399764.86058033, 396608.14952718,
                           393428.59970289, 390226.10441876, 387000.5605972 , 383751.86915647,
                           380479.93543002, 377184.66962277, 373865.987307  , 370523.80996025,
                           367158.06554728, 363768.68914774, 360355.62363063, 356918.82037527,
                           353458.24003717, 349973.85335477, 346465.64199026, 342933.59939333,
                           339377.73167143, 335798.05844264, 332194.61363709, 328567.44619854,
                           324916.62061501, 321242.21716943, 317544.33173307, 313823.07479591,
                           310078.5691835 , 306310.94544452, 302520.333018  , 298706.84365342,
                           294870.54050037, 291011.38048121, 287129.10617   , 283223.0397815 ,
                           279291.67893546, 275331.86216329, 271336.8947806 , 267291.69538703,
                           263150.27417383, 258674.41967721])


    θw_init_c = np.array([314.19261588, 311.67728772, 309.93547279, 308.72233543,
                           308.03232151, 307.998748  , 307.96492442, 307.93084864,
                           307.89651864, 307.86193236, 307.82708769, 307.79198253,
                           307.75661469, 307.72098197, 307.68508216, 307.64891296,
                           307.6124721 , 307.57575721, 307.53876594, 307.50149587,
                           307.46394456, 307.42610954, 307.38798829, 307.34957826,
                           307.31087686, 307.27188149, 307.23258948, 307.19299814,
                           307.15310474, 307.11290652, 307.07240068, 307.03158439,
                           306.99045476, 306.9490089 , 306.90724386, 306.86515665,
                           306.82274426, 306.78000363, 306.73693167, 306.69352525,
                           306.64978121, 306.60569633, 306.56126739, 306.5164911 ,
                           306.47136416, 306.42588321, 306.38004486, 306.33384558,
                           306.28172309, 306.18539579])


    θl_init_c = np.array([306.36699766, 306.18324625, 306.05428082, 305.96322827,
                           305.8985634 , 305.84861468, 305.79828719, 305.74757738,
                           305.69648169, 305.64499651, 305.59311821, 305.5408431 ,
                           305.48816747, 305.43508758, 305.38159962, 305.32769978,
                           305.27338417, 305.21864887, 305.16348993, 305.10790333,
                           305.05188502, 304.99543091, 304.93853685, 304.88119865,
                           304.82341206, 304.76517281, 304.70647656, 304.64731893,
                           304.58769549, 304.52760178, 304.46703326, 304.40598537,
                           304.34445349, 304.28243297, 304.21991909, 304.15690708,
                           304.09339216, 304.02936947, 303.9648341 , 303.89978113,
                           303.83420555, 303.76810233, 303.7014664 , 303.63429261,
                           303.56657581, 303.49831077, 303.42949223, 303.36011488,
                           303.29017338, 303.21979328])
    
    Ac, krw_c, klw_c, Cw_c, s_c = 3.65, 500, 1000, 3532500, 0.004
    condenser = HeatExchanger(Nc, Ac, krw_c, klw_c, Cw_c, s_c, h_init_c, θw_init_c, θl_init_c)

    Vd, eta_v, eta_s = 0.0002, 0.8, 0.7
    compressor = Compressor(Vd, eta_v, eta_s)

    Kv = 0.25
    expansion_valve = EV(Kv)

    U_init, Kp, Ki = 0.2542054339066616, 0.01, 1.0
    pi_controller = PIController(U_init, Kp, Ki)

    system = System(evaporator, condenser, separator, compressor, expansion_valve, pi_controller)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8

    gs = GridSpec(7, 2, figure=fig, width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])
    ax9 = fig.add_subplot(gs[4:, :])
    
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    plotlogph(ax9, refrigerant)

    lines_ev_w, = ax1.plot(np.linspace(0, 1, Nv), evaporator.θw - 273.15, label='Wall Temperature')
    lines_ev_l, = ax1.plot(np.linspace(0, 1, Nv), evaporator.θl - 273.15, label='Liquid Temperature')
    lines_ev_r, = ax1.plot(np.linspace(0, 1, Nv), evaporator.θr, label='Refrigerant Temperature')
    lines_cond_w, = ax2.plot(np.linspace(0, 1, Nc), condenser.θw - 273.15, label='Wall Temperature')
    lines_cond_l, = ax2.plot(np.linspace(0, 1, Nc), condenser.θl - 273.15, label='Liquid Temperature')
    lines_cond_r, = ax2.plot(np.linspace(0, 1, Nc), condenser.θr, label='Refrigerant Temperature')
    lines_pc, = ax3.plot([], [], label='p$_c$')
    lines_pv, = ax3.plot([], [], label='p$_v$')
    lines_U, = ax4.plot([], [], label='U')
    lines_SH, = ax5.plot([], [], label='SH')
    lines_m, = ax6.plot([], [], label='m')
    lines_EER, = ax7.plot([], [], label='EER')
    lines_Q0, = ax8.plot([], [], label='Q$_0$')
    lines_Qc, = ax8.plot([], [], label='Q$_c$')
    lines_Pcomp, = ax8.plot([], [], label='P$_{comp}$')

    line_logph, = ax9.plot([], [])

    ax1.set_ylabel('T / °C')
    ax1.set_xlabel('A/A$_{tot}$')
    ax2.set_xlabel('A/A$_{tot}$')
    ax2.set_ylabel('T / °C')
    ax3.set_xlabel('Time / s')
    ax3.set_ylabel('p / bar')
    ax3.set_yscale('log')
    ax4.set_xlabel('Time / s')
    ax4.set_ylabel('U / %')
    ax5.set_xlabel('Time / s')
    ax5.set_ylabel('SH / K')
    ax6.set_xlabel('Time / s')
    ax6.set_ylabel('$\dot{m}$ / kg/h')
    ax7.set_xlabel('Time / s')
    ax7.set_ylabel('EER')
    ax8.set_xlabel('Time / s')
    ax8.set_ylabel('kW')

    ax1.legend(frameon=False, fancybox=False)
    ax2.legend(frameon=False, fancybox=False)
    ax3.legend(frameon=False, fancybox=False)
    ax8.legend(frameon=False, fancybox=False)

    ax1.grid('True')
    ax2.grid('True')
    ax3.grid('True')
    ax4.grid('True')
    ax5.grid('True')
    ax6.grid('True')
    ax7.grid('True')
    ax8.grid('True')
    ax9.grid('True')

    ax1.set_title('Evaporator')
    ax2.set_title('Condenser')

    dt = 0.1
    t_span = [0, dt]

    po = [130223.88512533692]
    U = []

    ho_comp = []
    m = []
    ho_evap = []
    θo_evap_sec = []
    ho_cond = []
    θo_cond_sec = []
    pc = []
    pv = []
    ho_sep = []
    h_sep = []
    ho_ev = []
    SH = []
    Vl = []
    M_sep = []
    times = []
    Q0 = []
    Qc = []
    Pcomp = []
    EER = []
    
    
    f = f_slider.value
    m_evap_sec = m_evap_sec_slider.value
    θi_evap_sec = θi_evap_sec_slider.value
    m_cond_sec = m_cond_sec_slider.value
    θi_cond_sec = θi_cond_sec_slider.value
    SHsp = SHsp_slider.value
    
    i = 0
    while True:

        if control['stop_simulation']:
            plt.close()
            f_slider.value = 30.0
            m_evap_sec_slider.value = 1.0
            θi_evap_sec_slider.value = 5.0
            m_cond_sec_slider.value = 1.0
            θi_cond_sec_slider.value = 30.0
            SHsp_slider.value = 10.0
            break
        
        f = adjust_slider_value(f_slider, f, 0.1)
        m_evap_sec = adjust_slider_value(m_evap_sec_slider, m_evap_sec, 0.01)
        θi_evap_sec = adjust_slider_value(θi_evap_sec_slider, θi_evap_sec, 0.1)
        m_cond_sec = adjust_slider_value(m_cond_sec_slider, m_cond_sec, 0.01)
        θi_cond_sec = adjust_slider_value(θi_cond_sec_slider, θi_cond_sec, 0.1)
        SHsp = adjust_slider_value(SHsp_slider, SHsp, 0.1)

        def fun(po):
            compressor.model(f, evaporator.h[-1], po[-1], separator.p, refrigerant)
            expansion_valve.model(U_init, separator.ho, separator.p, po[-1], refrigerant)
            return np.array([compressor.m - expansion_valve.m])

        po += [fsolve(fun, np.array(po[-1]))[0]]

        system.solve(t_span, po[-1],
                     m_evap_sec, p_evap_sec, θi_evap_sec + 273.15,
                     m_cond_sec, p_cond_sec, θi_cond_sec + 273.15,
                     refrigerant, liquid_evap, liquid_cond)

        Tsat = PropsSI('T', 'P', po[-1], 'Q', 1.0, refrigerant)
        hsp = PropsSI('H', 'P', po[-1], 'T', Tsat + SHsp, refrigerant)
        U_init = pi_controller.model(dt, (hsp - evaporator.h[-1]) * 1e-5)
        U += [U_init * 100]
        SH += [PropsSI('T', 'P', po[-1], 'H', evaporator.h[-1], refrigerant) - Tsat]
        
        ρ_sep = PropsSI('D', 'P', separator.p, 'H', separator.h, refrigerant)
        ρ_sep_l = PropsSI('D', 'P', separator.p, 'Q', 0.0, refrigerant)
        ρ_sep_v = PropsSI('D', 'P', separator.p, 'Q', 1.0, refrigerant)
        M_sep += [V * ρ_sep]
        Vl += [V * (ρ_sep - ρ_sep_v) / (ρ_sep_l + ρ_sep_v)]

        ho_comp += [compressor.ho]
        m += [compressor.m * 3600]
        ho_cond += [condenser.h[-1]]
        θo_cond_sec += [condenser.θl[-1]]
        ho_evap += [evaporator.h[-1]]
        θo_evap_sec += [evaporator.θl[-1]]
        pc += [separator.p * 1e-5]
        pv += [po[-1] * 1e-5]
        h_sep += [separator.h]
        ho_sep += [separator.ho]
        ho_ev += [expansion_valve.ho]
        times += [i * dt]
        Q0 += [compressor.m * (ho_evap[-1] - ho_ev[-1]) / 1000]
        Qc += [compressor.m * (ho_comp[-1] - ho_cond[-1]) / 1000]
        Pcomp += [compressor.m * (ho_comp[-1] - ho_evap[-1]) / 1000]
        EER += [Q0[-1] / Pcomp[-1]]
        
        nplot = 5000

        if np.mod(i, 10) == 0:
            lines_ev_w.set_ydata(evaporator.θw - 273.15)
            lines_ev_l.set_ydata(evaporator.θl - 273.15)
            lines_ev_r.set_ydata(evaporator.θr - 273.15)
            lines_cond_w.set_ydata(condenser.θw - 273.15)
            lines_cond_l.set_ydata(condenser.θl - 273.15)
            lines_cond_r.set_ydata(condenser.θr - 273.15)
            lines_pc.set_data(times[-nplot:], pc[-nplot:])
            lines_pv.set_data(times[-nplot:], pv[-nplot:])
            lines_U.set_data(times[-nplot:], U[-nplot:])
            lines_SH.set_data(times[-nplot:], SH[-nplot:])
            lines_m.set_data(times[-nplot:], m[-nplot:])
            lines_EER.set_data(times[-nplot:], EER[-nplot:])
            lines_Q0.set_data(times[-nplot:], Q0[-nplot:])
            lines_Qc.set_data(times[-nplot:], Qc[-nplot:])
            lines_Pcomp.set_data(times[-nplot:], Pcomp[-nplot:])
            line_logph.set_data([ho_evap[-1], ho_comp[-1], ho_cond[-1], ho_sep[-1], ho_ev[-1], ho_evap[-1]],
                                [pv[-1], pc[-1], pc[-1], pc[-1], pv[-1], pv[-1]])

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            ax4.relim()
            ax4.autoscale_view()
            ax5.relim()
            ax5.autoscale_view()
            ax6.relim()
            ax6.autoscale_view()
            ax7.relim()
            ax7.autoscale_view()
            ax8.relim()
            ax8.autoscale_view()
            ax9.relim()
            ax9.autoscale_view()

            fig.suptitle(f'Time {i * dt} s')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        i += 1