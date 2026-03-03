import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import ib_capture_bias

class PolyAModel_IB():
    '''
    individual based model
    - models poly(A) tail dynamcis of each transcript
    - different degradation mechanisms:
        - PD:  poly(A) tail length dependent
        - MD:  miRNA mediated decay (poly(A) tail independent)
        - PMD: miRNA tail shortening + poly(A) tail length dependent decay
    '''

    def __init__(self):
        
        self.name = "PolyAModel_IB"
        
        self.P_max = 100
        self.P_min = 10.0
        self.K_b = 15.85
        self.n = 2
        self.S_depth = 3.8e6
        self.L_mean_kb = 1.94

    ''' ----------------------------------------------- 
        functions: tail dynamics, regulator activity, decay probability 
        -------------------------------------------------
    '''
    @staticmethod
    def polyA_tail_dynamic(t, P, P_max, k_p, k_d, rep):
        """ dP/dt = k_p * cpa(t) * (P_max - P) - k_d * rep(t) * P """
        #polya = k_p * cpa(t) * (P_max - P)
        polya = k_p * P * (1 - P/P_max)
        #polya = k_p
        deaden = k_d * rep(t) * P
        return polya - deaden

    @staticmethod
    def polyA_decay_prob(P, P_min, s=0.9):
        """ logistic function between 0 and 1 with midpoint at P_min"""
        #return 1 / (1 + np.exp(s * (P - P_min)))
        #return np.maximum(P_min - P, 0.0)
        return np.log1p(np.exp(P_min - P))

    @staticmethod
    def regulator_activity(t, t_on=3, s=10):
        ''' increase from 0 to 1 between t_on and t_off '''
        t = np.asarray(t)
        return 1 / ( 1 + (t_on/t)**s)

    @staticmethod
    def cpa_timing(t, t_on, t_off=0.5):
        ''' linear increase from 0 to 1 between t_on and t_off '''
        t = np.asarray(t)
        slope = 1/(t_off - t_on)
        return np.where(t <= t_on, 0.0,
               np.where(t <= t_off, slope * (t - t_on), 1.0))
    
    def calc_P0(self, y0, ymax):
        fb = y0/ymax
        return ((self.K_b**self.n * fb)/(1-fb))**(1/self.n)
    
    ''' ----------------------------------------------- 
        Simulation 
        -----------------------------------------------
    '''
    def simulate(self,
                 gene_id,
                 n_transcripts,
                 P0_mean=15.1,
                 P0_std= 1,
                 k_p = 1, 
                 k_d = 5.0,
                 t_deg = 3,
                 dt=0.01,
                 t_end=6.0,
                 deg_mechanism="PD",
                 seed=10):

        # initialize
        rng = np.random.default_rng(seed)
        #P = rng.normal(P0_mean, P0_std, n_transcripts) # normal
        P = rng.gamma(P0_mean, P0_std, n_transcripts)   # gamma
        P = np.clip(P, 0, self.P_max)

        time = np.arange(0, t_end+dt, dt)

        # regulator and CPA functions
        reg_t = np.linspace(0, t_end, 600)
        rep = self.regulator_activity(reg_t, t_deg)
        #cpa = self.cpa_timing(reg_t, self.t_cpa)

        rep_fn = interp1d(reg_t, rep)
        #cpa_fn = interp1d(reg_t, cpa)

        # storge arrays
        alive = np.ones(n_transcripts, bool) # store transcript state : alive = True, degradad = False
        P_traj = np.full((len(time), n_transcripts), np.nan) # store for poly(A) tail lengths over time

        # simulation loop
        for i, t in enumerate(time):

            P_t = P.copy()
            P_t[~alive] = np.nan # set degraded transcripts NAN
            P_traj[i, :] = P_t  # update tail lengths for current time step

            # update poly(A) tail length for alive transcripts - Euler method
            dP_dt = self.polyA_tail_dynamic(t, P[alive], self.P_max,
                                         k_p, k_d, rep_fn)
            P[alive] += dt * dP_dt

            # stochastic decay
            if deg_mechanism == "PD":
                # degradation probability dependent on P tail length
                # short tails -> high degradation probability (~1), long tails -> low prob (~0)
                prob = dt * self.polyA_decay_prob(P[alive], self.P_min) * rep_fn(t) # miRNA activity
                survive =  rng.random(alive.sum()) > prob

            elif deg_mechanism == "MD":
                # degadation probability dependent on miRNA activity
                # -> miRNA miR-430 is responsible for degradation of many maternal transcripts
                # -> maternal transcripts are otherwise stable
                target_prob = dt * rep_fn(t)
                targeted = (rng.random(alive.sum()) < target_prob) # stochastic targeting by miRNA
                survive = np.ones(alive.sum(), dtype=bool)

                if targeted.sum() > 0:
                    # decay probability only for targeted transcripts
                    prob = dt * self.polyA_decay_prob( P[alive][targeted], self.P_min )

                    # stochastic decay only for targeted ones
                    survive[targeted] = ( rng.random(targeted.sum()) > prob )

                #prob = dt * self.polyA_decay_prob(P[alive], self.P_min)

            elif deg_mechanism == "MPD":
                # miRNA shortens poly(A) first to P_min -> "targeting"
                P_alive = P[alive].copy()
                target_prob = dt * rep_fn(t)
                shorten =  (rng.random(alive.sum()) < target_prob) & (P_alive > self.P_min) #  avoid increase in tail length if P < P_min
                P_alive[shorten] = self.P_min
                P[alive] = P_alive
                # degradation probability depends on P tail length
                prob = dt * self.polyA_decay_prob(P[alive], self.P_min)
                # uniform distribution [0,1)
                survive =  rng.random(alive.sum()) > prob

            # update transcript state
            temp = alive.copy()
            temp[alive] = survive
            alive = temp

        ds = ib_capture_bias.simulate_capture_bias(P_traj, time, gene_id)

        #return time, P_traj
        return ds
    
    ''' ----------------------------------------------- 
        Plot functions 
        ----------------------------------------------- 
    '''
    def plot_functions(self, t_deg, t_cpa=0):
        ''' Plot regulator and probability functions'''

        def polyA_capture_prob(P, K_b, n=2):
            return (P**n) / (K_b**n + P**n)
        
        import matplotlib.pyplot as plt
        P = np.linspace(0, 100, 500)
        time = np.linspace(0, 5, 100)

        fig, ax = plt.subplots(1,3, figsize=(8, 2.5))

        ax[0].plot(P, self.polyA_decay_prob(P, self.P_min), c="darkred")
        ax[0].axvline(x=self.P_min, ymin=0, ymax=1, ls="dashed", c="grey")

        ax[1].plot(P, polyA_capture_prob(P, self.K_b), c="purple")
        ax[1].axvline(x=self.K_b, ymin=0, ymax=1, ls="dashed", c="grey")
        
        ax[2].plot(time, self.regulator_activity(time, t_deg), c="k", label="miRNA")
        #ax[2].plot(time, self.cpa_timing(time, self.t_cpa), c="g", label="CPA")

        ax[0].set(title="degradation probability",xlabel="Poly(A) tail length (nt)",)
        ax[1].set(title="oligo(dT) binding affinity",xlabel="Poly(A) tail length (nt)")
        ax[2].set(title="regulator activity",xlabel="time [hpf]")
        ax[2].legend(loc=(1.01, 0.3), frameon=False)
        plt.tight_layout()
        plt.savefig(f"figures/polyA_functions_ib-model_tdeg_{t_deg}.png")
        plt.show()
        

