import numpy as np
import click
import jax
import jax.numpy as jnp
import pandas as pd
import os
import arviz as az
import xarray as xr


class PolyAModel_mean():
    '''
    Models transcript abundance and mean polyA tail length
    polyA selection bias is applied afterwards
    '''

    global polyA_decay_prob
    global regulator_activity
    global transcript_length

    def __init__(self,
                L_frag_kb = 0.1 * 2,     # White et al.
                L_mean_kb = 1.94218,     # White et al.
                S_depth = 3.8e6,         # White et al.

                P_mean = 100.0,       # Subtelny et al (2014) 

                K_b = 15.85,           #  fitted, Meijer (2007)
                n = 2.0,               # fitted, Meijer (2007)
                P_min = 10.0,          # Lorenzo-Orts & Pauli (2024)
                k_d = 5.0,             # free parameter
                k_p = 1.0              # free parameter
                ):


        self.name = "PolyAModel_mean_approx"

        self.L_frag_kb = L_frag_kb
        self.S_depth = S_depth
        self.scaling_factor = 1e6 / (S_depth / L_mean_kb)  # TPM scaling factor: 1e6 / sum(RPK/L)

        self.K_b = K_b
        self.n = n

        self.P_min = P_min
        self.P_mean = P_mean
        self.k_d = k_d
        self.k_p = k_p


    def polyA_decay_prob(P, P_min):
        '''
        biological degradation:
        short polyA -> high decay
        '''
        return jax.nn.softplus(P_min - P)

    def polyA_enrichment_step(self, T_true, P, K_b, n):
        '''
        sequencing capture bias:
        long polyA -> more oligo(dT) captures
        '''
        f_binding = P**n / (K_b**n + P**n)
        return T_true * f_binding
    
    def calc_P0(self, y0, ymax, ):
        fb = y0/ymax
        return ((self.K_b**self.n * fb)/(1-fb))**(1/self.n)

    def transcript_length(gene_id):
        """return transcript length and mean length of full dataset [kb]"""
        tpm = pd.read_csv("data/tpms_white_pauli_JN_BK.csv")
        tpm = tpm[tpm["transcript_is_canonical"] == 1.0]
        L_transcript = tpm[tpm["ensembl_gene_id"] == gene_id].transcript_length.item() /1000
        return L_transcript

    def TPM(self, n_transcripts, L_transcript_kb, scaling_factor):
        '''
        n_transcripts : absolute transcript abundance
        TPM = (reads/L) * scaling_factor 
        '''
        RPK = n_transcripts/L_transcript_kb
        TPM = RPK * scaling_factor
        return TPM

    @staticmethod
    def regulator_activity(t, t_on=3, s=10):
        t = np.asarray(t)
        s = 10
        return t**s / ( t**s + t_on**s)

    
    @staticmethod
    def cpa_timing(t, t_on=0, t_off=1.0):
        t = np.asarray(t)
        slope = 1/(t_off - t_on)
        return np.where(t <= t_on, 0.0,
                np.where(t <= t_off, slope * (t - t_on), 1.0))
    
    
    def _rhs_model(t, y, x_in, k_p, k_d, P_min, P_mean):
        ''' 
        P: median Poly(A) tail length   [nt]
        H: degradation hazard
        k_p: polyadenylation rate       [1/h]
        k_p: deadenylation rate         [1/h]
        P_mean: mean polyA tail length  [nt]
        P_min: polyA tail degradation threshold  [nt]
        '''
        P, H = y
        rep = x_in.evaluate(t)

        h = polyA_decay_prob(P, P_min) * rep
        dPdt =  k_p * P * (1 - P/P_mean) - (k_d * rep )  * P  ## logistic
        #dPdt =  k_p * (P_mean - P) - (k_d * rep )  * P 
        #dPdt =  k_p - (k_d * rep)  * P    # no scaled k_p
        dHdt = h
        return dPdt, dHdt 
    
    def _post_processing(self, results, time, interpolation):
        '''Apply poly(A) sequencing bias and conversion to TPM'''

        results["T"] = self.T0 * jnp.exp(-results["H"])
        T_seq = self.polyA_enrichment_step(results["T"], results["P"], self.K_b, self.n)

        results["TPM_true"] = self.TPM(results["T"], self.L_transcript_kb, self.scaling_factor)
        results["TPM_biased"] = self.TPM(T_seq, self.L_transcript_kb, self.scaling_factor)

        results["miRNA"] = jax.vmap(interpolation.evaluate)(time)

        return results


    def simulate(self, gene_id, seed, eval=False, plot=True, kernel="nuts", gene_name = ""):

        from pymob.simulation import SimulationBase
        from pymob.sim.config import DataVariable
        from pymob.sim.parameters import Param
        from pymob.solvers.diffrax import JaxSolver

        # --- prepare pymob
        sim = SimulationBase()

        sim.config.case_study.name = f"{self.name}"+"_White_tails"
        sim.config.case_study.scenario = f"{gene_id}"
        sim.config.simulation.x_dimension = "time"

        output = os.getenv("RESULTS_DIR", "./results")
        os.makedirs(output, exist_ok=True)
        gene_output_dir = os.path.join(output, sim.config.case_study.name , sim.config.case_study.scenario)

        if os.path.exists(os.path.join(gene_output_dir, "numpyro_posterior.nc")):
            print(f"[SKIP] Gene {gene_id} already processed — skipping.")
            return
        
        os.makedirs(gene_output_dir, exist_ok=True)
        sim.config.case_study.output_path = gene_output_dir
        sim.config.create_directory("scenario", force=True)
        
        # obsvervation data
        obs = self.prepare_data(gene_id=gene_id)
        sim.observations = obs
        sim.model = self._rhs_model
        sim.solver = JaxSolver
        sim.solver_post_processing = self._post_processing    
        sim.config.simulation.n_ode_states = 2
        sim.config.simulation.seed = seed

        # Free parameter
        sim.config.model_parameters.k_p = Param(value=self.k_p, free=True, prior=f"lognorm(scale={self.k_p}, s=1)")
        sim.config.model_parameters.k_d = Param(value=self.k_d,  free=True,  prior=f"lognorm(scale={self.k_d}, s=1)")
        sim.config.model_parameters.P_mean = Param(value=self.P_mean, free=False, prior=f"gamma(a={self.P_mean}, scale=1)" )
        sim.config.model_parameters.sigma_y = Param(value=0.1, free=True , prior="lognorm(scale=0.5, s=0.5)", min=1e-3, max=1)
        
        # Fixed
        sim.config.model_parameters.P_min = Param(value=self.P_min, free=False)

        # State variables 
        sim.config.data_structure.P = DataVariable(dimensions=("time",), observed=True)
        sim.config.data_structure.H = DataVariable(dimensions=("time",), observed=False)

        sim.config.data_structure.T = DataVariable(dimensions=("time",), observed=False)
        sim.config.data_structure.TPM_true = DataVariable(dimensions=("time",), observed=False)
        sim.config.data_structure.TPM_biased = DataVariable(dimensions=("time",), observed=False) # observed data
        sim.config.data_structure.miRNA = DataVariable(dimensions=("time",), observed=False)

        # input data - x_in
        sim.config.simulation.x_in = ["miRNA=miRNA"]
        sim.model_parameters["x_in"] = sim.parse_input(input="x_in", reference_data=sim.observations, drop_dims=[])

        y_max = sim.observations.TPM_biased.max().item()
        y_0 = sim.observations.TPM_biased.sel(time=0).item()

        self.L_transcript_kb = transcript_length(gene_id)
        self.T0 = (y_max * self.L_transcript_kb) / self.scaling_factor  # "true" read count
        print("SF ", self.scaling_factor)
        print("T0: ", self.T0)

        P0 = self.calc_P0(y_0, y_max)
        sim.config.simulation.y0 = [f"P={P0}", "H=0.0"]
        sim.model_parameters["y0"] = sim.parse_input("y0", sim.observations, drop_dims=("time"))

        print(sim.model_parameters["y0"])

        #sim.config.error_model.TPM_biased = "normal(loc=TPM_biased, scale=sigma_y)"
        sim.config.error_model.P = "normal(loc=P, scale=sigma_y)"
        sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

        print(sim.model_parameters["parameters"] )

        if eval:
            sim.coordinates["time"]= np.linspace(0, 6, 600)
            sim.dispatch_constructor()
            evaluator = sim.dispatch()
            evaluator()
            results = evaluator.results
            if plot:
                self.plot_eval_results(results, obs, out=gene_output_dir, title=f"eval_"+gene_id )
            return results, 
        
        sim.set_inferer("numpyro")
        sim.config.jaxsolver.diffrax_solver = "Dopri5"
        sim.config.jaxsolver.atol = 1e-12
        sim.config.jaxsolver.rtol = 1e-10
        sim.config.inference_numpyro.kernel = kernel
        sim.config.jaxsolver.throw_exception = False

        sim.config.inference_numpyro.gaussian_base_distribution = True
        sim.config.inference_numpyro.init_strategy= "init_to_median"

        sim.config.inference_numpyro.svi_iterations = 3000
        sim.config.inference_numpyro.svi_learning_rate = 0.01

        sim.config.inference_numpyro.chains = 4
        sim.config.inference_numpyro.draws = 2000
        sim.config.inference_numpyro.warmup = 1000

        sim.dispatch_constructor()
        sim.prior_predictive_checks(pred_mode="draws",)
    
        try:
            sim.inferer.run()
        except Exception as e:
            print(f"[ERROR] Gene {gene_id} failed: {e}")
            return 

        sim.dispatch_constructor()

        sim.inferer.store_results()
        sim.posterior_predictive_checks(pred_mode="mean+hdi", pred_hdi_style={"color": "#7034b1", "alpha": .15})
        sim.report()
        #sim.save_observations(force=True)
        sim.config.save(force=True)

        results = sim.inferer.idata
        if plot: 
            self.plot_eval_results1(results.posterior_model_fits, obs, out=gene_output_dir, title=f"{gene_id} ({gene_name})")
        self.gene = gene_id
        
        return

    def plot_eval_results1(self, res, obs, out, title=""):

        from matplotlib import pyplot as plt
        import arviz as az

        fig, ax = plt.subplots(2,1, figsize=(8, 5), height_ratios=[1,1])

        hdi_true = az.hdi(res, 0.95).TPM_true
        hdi_bias = az.hdi(res, 0.95).TPM_biased
        hdi_P = az.hdi(res, 0.95).P

        res = res.mean(dim=("chain", "draw"))

        ax[1].plot(res.time, res.P, lw=2, c="darkgreen")
        ax[0].plot(res.time, res.TPM_true, label="debiased")
        ax[0].plot(res.time, res.TPM_biased, c="r", label="polyA+ biased")

        ax[0].plot(obs.time, obs.TPM_biased, ".", c="k", alpha=0.8, label="White et al. (2017)")
        ax[1].plot(obs.time, obs.P, "x", c="k", label="Subtelny et al. (2014)")

        ax[0].fill_between(hdi_true.time, *hdi_true.values.T, color="blue", alpha=0.05,)
        ax[0].fill_between(hdi_bias.time, *hdi_bias.values.T, color="r", alpha=0.05,)
        ax[1].fill_between(hdi_P.time, *hdi_P.values.T, color="darkgreen", alpha=0.05,)

        ax[1].axhline(y=12, xmin=0, xmax=6, label="P_min", ls="dashed", c="grey")
        ax[1].axhline(y=20, xmin=0, xmax=6, label="K_b", ls="dashdot", c="grey")

        ax[1].set(title="Poly(A) tail length", xlabel="time [hpf]", ylabel="[nt]")
        ax[0].set(title=title, xlabel="time [hpf]", ylabel="TPM")

        ax[1].legend(loc=(1.01, 0.2), frameon=False)
        ax[0].legend(loc=(1.01, 0.3), frameon=False)

        plt.tight_layout()
        if out != None:
            plt.savefig(f"{out}/polyA_mean_pp_{title}.png")
        plt.show()
    

    def plot_eval_results(self, res, obs, out, title=""):

        from matplotlib import pyplot as plt
        import arviz as az

        fig, ax = plt.subplots(2,1, figsize=(8, 5), height_ratios=[1,1])

        ax[1].plot(res.time, res.P, lw=2, c="darkgreen")

        ax[0].plot(res.time, res.TPM_true, label="true")
        ax[0].plot(res.time, res.TPM_biased, c="r", label="biased")
        ax[0].plot(obs.time, obs.TPM_biased, "o", c="k", alpha=0.6, label="observed")

        ax[1].axhline(y=self.P_min, xmin=0, xmax=6, label="P_min", ls="dashed", c="grey")
        ax[1].axhline(y=self.K_b, xmin=0, xmax=6, label="K_b", ls="dashdot", c="grey")

        ax[1].set(title="Poly(A) tail length", xlabel="time [hpf]", ylabel="[nt]")
        ax[0].set(title=title, xlabel="time [hpf]", ylabel="TPM")

        ax[1].legend(loc=(1.05, 0.1))
        ax[0].legend(loc=(1.05, 0.1))

        plt.tight_layout()
        if out != None:
            plt.savefig(f"{out}/polyA_mean_pp_{title}.png")
        plt.show()


    def plot_functions(self, t_deg, t_cpa=0):
        ''' Plot regulator and probability functions'''

        def polyA_capture_prob(P, K_b, n=10):
            return (P**n) / (K_b**n + P**n)
        
        import matplotlib.pyplot as plt
        P = np.linspace(0, 100, 500)
        time = np.linspace(0, 5, 100)

        fig, ax = plt.subplots(1,3, figsize=(8, 2.5))

        ax[0].plot(P, self.polyA_decay_prob(P), c="darkred")
        ax[0].axvline(x=self.P_min, ymin=0, ymax=1, ls="dashed", c="grey")

        ax[1].plot(P, polyA_capture_prob(P, self.K_b), c="purple")
        ax[1].axvline(x=self.K_b, ymin=0, ymax=1, ls="dashed", c="grey")
        
        ax[2].plot(time, self.regulator_activity(time, t_deg), c="k", label="miRNA")
        ax[2].plot(time, self.cpa_timing(time, t_cpa), c="g", label="CPA")

        ax[0].set(title="degradation probability",xlabel="Poly(A) tail length [nt]",)
        ax[1].set(title="oligo(dT) binding affinity",xlabel="Poly(A) tail length [nt]")
        ax[2].set(title="regulator activity",xlabel="time [hpf]")
        ax[2].legend(loc=(1.01, 0.3))
        plt.tight_layout()


    def prepare_data(self, gene_id="ENSDARG00000040266"):
        import xarray as xr

        data = xr.load_dataset("data/white_dataset_mean.nc").sel(time=slice(0, 6))
        tails = pd.read_csv("data/tail_lengths.csv")
        P = tails.loc[tails["GeneID"] == gene_id]
        t0 = np.linspace(0, 6, 31)
        t_tails = [2, 4, 6]
        t1 = np.array(data.time.values)
        t = np.sort(np.unique(np.append(t0, t1)))
        
        tails_i = [P["2 hpf"].item(), P["4 hpf"].item(), P["6 hpf"].item()]
        p_obs = xr.Dataset(data_vars= dict(P = ("time", tails_i)), coords=dict(time=t_tails))

        obs = data.sel(ensembl_gene_id=gene_id)
        obs = obs.interp(time=t, method="linear") 
        obs = obs.reindex(time=np.sort(np.unique(np.concatenate([data.time.values, t]))))
        
        r = regulator_activity(t, t_on=3, s=10) # linear: t_on = 2.25
        rep = xr.Dataset(data_vars= dict(miRNA = ("time", r)), coords=dict(time=t ))
        #rep_on_obs = rep.interp(time_rep=obs.time).drop_vars("time_rep")
        #p_on_obs = p_obs.interp(t_tails=obs.time, method="quadratic").drop_vars("t_tails")

        data = xr.merge([obs, rep, p_obs])
        data = data.rename({"y":"TPM_biased"})
        return data
    