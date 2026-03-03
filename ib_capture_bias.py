import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
#  capture functions
# -------------------------------------------------------------------

def polyA_capture_prob(P, K_b, n=2):
    """
    Hill-type capture bias.
    P : poly(A) length
    K_b : half-saturation (K_b)
    """
    return (P**n) / (K_b**n + P**n)

def transcript_length(gene_id):
    """return transcript length and mean length of full dataset [kb]"""
    tpm = pd.read_csv("data/tpms_white_pauli_JN_BK.csv")
    tpm = tpm[tpm["transcript_is_canonical"] == 1.0]
    L_transcript = tpm[tpm["ensembl_gene_id"] == gene_id].transcript_length.item() /1000
    return L_transcript


def TPM(n_transcripts, L_transcript_kb, scaling_factor):
    '''
    n_transcripts : absolute transcript abundance
    TPM = (reads/L) * scaling_factor 
    '''
    RPK = n_transcripts/L_transcript_kb
    TPM = RPK * scaling_factor
    return TPM


# -------------------------------------------------------------------
#  capture-bias simulation
# -------------------------------------------------------------------

def simulate_capture_bias(P_traj,
                          time,
                          gene_id,
                          K_b=15.85,
                          L_frag_kb=0.1,
                          L_transcript_kb=2.5,
                          L_mean_kb = 1.94,
                          S_depth=3.8e6,
                          P_min=10.0,
                          seed=10):
    """
    Simulate poly(A)-dependent sequencing bias.

    Parameters
    ----------
    P_traj :          Poly(A) tail lengths from simulation (NaN = degraded)
    time :            Time points
    K_b :             Poly(A) length where capture is 50%
    L_frag_kb :       Fragment size (kb) used for read normalization
    L_transcript_kb : Transcript length (kb)
    S_depth :         Sequencing depth for TPM calculation
    P_min :           Minimum tail length 
    seed :            Random seed for reproducibility
    """

    L_transcript_kb = transcript_length(gene_id)
    scaling_factor = 1e6/(S_depth / L_mean_kb)

    rng = np.random.default_rng(seed)

    # count true transcripts per time point
    T_true = np.sum(~np.isnan(P_traj), axis=1)

    # capture probability per transcript
    P_cap_i = P_traj.copy()
    prob = polyA_capture_prob(P_cap_i, K_b)

    rand = rng.random(prob.shape)
    captured = rand < prob    # True, when oligo(dT) pearls bind to poly(A) tail

    P_captured = np.where(captured, P_cap_i, np.nan)  # set uncaptured transcripts NAN
    T_seq = np.sum(~np.isnan(P_captured), axis=1)  # count sequenced transcripts

    # convert to reads
    #reads_true = read_counts(T_true, L_transcript_kb, L_frag_kb)
    #reads_seq  = read_counts(T_seq,  L_transcript_kb, L_frag_kb)

    # TPM
    tpm_true = TPM(T_true, L_transcript_kb, scaling_factor)
    tpm_seq  = TPM(T_seq,  L_transcript_kb, scaling_factor)

    # create xrDataset for results
    ds = xr.Dataset(
        data_vars=dict(
            P_true     = (["time", "transcript_id"], P_traj),
            P_captured = (["time", "transcript_id"], P_captured),
            T_true     = (["time"], T_true),
            T_seq      = (["time"], T_seq),
            TPM_true   = (["time"], tpm_true),
            TPM_seq    = (["time"], tpm_seq),
        ),
        coords=dict(
            time=time,
            transcript_id=np.arange(P_traj.shape[1]),
        ),
    )

    ds.to_netcdf(f"results/{gene_id}_ib_model_fit.nc")

    return ds


# -------------------------------------------------------------------
#  plotting
# -------------------------------------------------------------------

def plot_capture_results(ds, gene_id, K_b=15.85, P_min=10):
    """
    Plot summary of:
    - mean poly(A)
    - true vs captured TPM
    """
    P_mean = ds.P_true.mean("transcript_id")
    P_std  = ds.P_true.std("transcript_id")

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 5))

    #data = xr.load_dataset("data/white_dataset_mean.nc")
    data = xr.load_dataset("data/genes_tpms_white_pauli_JN_BK_mean_wSource.nc").sel(source="White et al.")
    obs = data.sel(ensembl_gene_id=gene_id).sel(time=slice(0, 6)) 

    # poly(A) summary
    ax1.plot(ds.time, P_mean, c="darkgreen")
    ax1.fill_between(ds.time, P_mean - P_std, P_mean + P_std, alpha=0.2, color="green")
    #ax1.axhline(K_b, color="grey", ls="dashdot", label=fr"$P_{{capture}}$")
    ax1.axhline(P_min, color="grey", ls="dashed",  label=fr"$P_{{min}}$")
    ax1.set(title="Mean poly(A) tail length", ylabel="nt", xlabel="time (hpf)",)
    ax1.legend(loc=(1.01, 0.3), frameon=False)

    # TPM
    ax2.plot(ds.time, ds.TPM_true, label="debiased", color="tab:blue")
    ax2.plot(ds.time, ds.TPM_seq,  label="polyA+ biased", color="red")
    ax2.plot(obs.time, obs.y,  "s", c="k", alpha=0.6, label="White et al.", )
    ax2.set(title=gene_id, xlabel="time (hpf)", ylabel="TPM")
    #ax2.axvline(x=3, ymin=0, ymax=1, ls="dashed", c="grey", label="ZGA")
    ax2.legend(loc=(1.01, 0.3), frameon=False)

    plt.tight_layout()
    plt.savefig(f"figures/ib_model/{gene_id}_polyA_capture_bias.png")
    #plt.show()
    plt.close()


def plot_tail_distribution(ds, gene_id):
        
        cmap = sns.color_palette("Dark2", n_colors=5)
        
        P_i_0 = ds.P_true.sel(time=0)
        P_i_2 = ds.P_true.sel(time=2)
        P_i_4 = ds.P_true.sel(time=4)
        #P_i_6 = ds.P_true.sel(time=6)

        fig, ax = plt.subplots(1, 3, figsize=(6,2.5), )#sharex="row")
        ax[0].hist(P_i_0, bins=30, color=cmap[0])
        ax[0].set(title="t = 0 hpf", xlabel="poly(A) tail length (nt)")

        ax[1].hist(P_i_2, bins=30, color=cmap[1])
        ax[1].set(title="t = 2 hpf", xlabel="poly(A) tail length (nt)")

        ax[2].hist(P_i_4, bins=30, color=cmap[2])
        ax[2].set(title="t = 4 hpf", xlabel="poly(A) tail length (nt)")

        plt.suptitle("poly(A) tail length")
        plt.tight_layout()
        plt.savefig(f"figures/ib_model/{gene_id}_polyA_tail_distribution_ib.png")
        #plt.show()
        plt.close()


def plot_tail_distribution2(ds, gene_id):
        cmap = sns.color_palette("Dark2", n_colors=5)
        
        P_i_0 = ds.P_true.sel(time=0)
        P_i_2 = ds.P_true.sel(time=2)
        P_i_4 = ds.P_true.sel(time=4)
        #P_i_6 = ds.P_true.sel(time=6)

        fig, ax = plt.subplots(figsize=(5,2.5), )
        ax.hist(P_i_0, bins=30, color=cmap[0], alpha=0.8, label= "0 hpf")
        ax.hist(P_i_2, bins=30, color=cmap[1],  alpha=0.8,label= "2 hpf")
        ax.hist(P_i_4, bins=30, color=cmap[2],  alpha=0.8, label= "4 hpf")
        ax.legend(frameon=False)
        ax.set(title="poly(A) tail length", xlabel="poly(A) tail length (nt)")

        plt.tight_layout()
        plt.savefig(f"figures/ib_model/{gene_id}_polyA_tail_distribution_ib2.png")
        #plt.show()
        plt.close()
