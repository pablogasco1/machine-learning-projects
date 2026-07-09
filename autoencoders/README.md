# Autoencoders — A Hands-On Course

A progressive, heavily-commented series of Jupyter notebooks that teaches **Autoencoders (AEs)** from
first principles up to the variants and problems you'll be asked about in a **consulting / ML
interview**. Each notebook is self-contained but they are designed to be read **in order**.

Built with **PyTorch**, runs **on CPU**. Early notebooks are fully **offline** (bundled data); the
image/generative notebooks download **MNIST / Fashion-MNIST** once via `torchvision` (cached locally,
git-ignored). Every notebook was executed and verified to run top-to-bottom.

The whole series rests on one idea, seen from seven angles:

> **Encoder → code (bottleneck) → decoder**, trained to reconstruct its input. Turn three knobs — the
> **bottleneck**, the **loss/regularizer**, and **what you feed in vs. ask out** — and a new capability
> appears each time.

## The path

| # | Notebook | What you'll learn | Key idea |
|---|----------|-------------------|----------|
| 01 | [`01_autoencoder_fundamentals.ipynb`](01_autoencoder_fundamentals.ipynb) | Encoder/bottleneck/decoder, reconstruction loss, the undercomplete AE, reading a 2-D latent space, and the deep link to **PCA** (linear AE = PCA). On 8×8 digits (offline). | *Learn to copy yourself through a narrow pipe → you're forced to learn structure.* |
| 02 | [`02_denoising_convolutional.ipynb`](02_denoising_convolutional.ipynb) | **Denoising AEs** (corrupt the input, target the clean original), the manifold-projection view, **convolutional** encoders/decoders, and **inpainting**. On MNIST. | *Don't copy the input — repair it. The seed idea behind diffusion models.* |
| 03 | [`03_anomaly_detection.ipynb`](03_anomaly_detection.ipynb) | The **flagship use case**: train on normal, use **reconstruction error as an anomaly score**, pick a threshold, evaluate with **ROC/PR-AUC**, and beat univariate rules on **correlation-breaking** anomalies. Benchmarked vs Isolation Forest / PCA / Mahalanobis. | *Learn "normal"; flag whatever doesn't fit.* |
| 04 | [`04_sparse_contractive_representation.ipynb`](04_sparse_contractive_representation.ipynb) | **Sparse** (L1 → few units fire, interpretable filters) and **contractive** (Jacobian → robust code) AEs, plus the **honest** story of AE **representation learning** (compression + robustness, not free accuracy). On Fashion-MNIST. | *Regularize the code instead of just shrinking it.* |
| 05 | [`05_variational_autoencoder.ipynb`](05_variational_autoencoder.ipynb) | **VAEs**: encode a *distribution*, the **reparameterization trick**, the **ELBO** (recon + KL), then **generate** new items, **interpolate**, and draw the 2-D latent **manifold**. VAE vs plain-AE sampling. On Fashion-MNIST. | *A KL term fills the latent space with no holes → you can sample it.* |
| 06 | [`06_sequence_lstm_autoencoder.ipynb`](06_sequence_lstm_autoencoder.ipynb) | **Sequence (LSTM) autoencoders** (seq2seq / EncDec-AD) for **time-series & sensor anomaly detection**: train on normal windows, score by reconstruction error, and **localize the fault in time**. | *Autoencode a whole window; abnormal dynamics reconstruct badly.* |
| 07 | [`07_wrapup_cheatsheet.ipynb`](07_wrapup_cheatsheet.ipynb) | The **interview map**: the autoencoder zoo, a **decision guide** (problem → AE), consulting problem→solution table, the equations on one page, and crisp answers to the common questions. | *Which autoencoder for which problem — and when to use none.* |

## A recurring, honest theme

Across notebooks 01, 03, 04 and 06 we repeatedly benchmark the neural autoencoder against **simpler
baselines** (PCA, Mahalanobis, Isolation Forest). The consistent, interview-ready lesson:

> **Match model complexity to the data's structure.** If the structure is linear / low-dimensional /
> Gaussian, a classical method is simpler and often just as good. Reach for the neural autoencoder when
> the structure is **nonlinear, high-dimensional, or structured** (images, sequences).

## How to run

From this folder, launch Jupyter and open the notebooks in order:

```bash
jupyter lab      # or: jupyter notebook
```

Or run one end-to-end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_autoencoder_fundamentals.ipynb
```

## Requirements

Covered by the repo's `requirements.txt`:

```
numpy · pandas · matplotlib · seaborn · scikit-learn · scipy · torch · torchvision
```

> Notebooks 02, 04, 05 download **MNIST / Fashion-MNIST** on first run into `autoencoders/data/`
> (git-ignored). Notebooks 01, 03, 06 are fully offline (bundled scikit-learn data / synthetic).

## Suggested study approach

1. **Read the markdown first**, then run each cell and confirm the output matches the explanation.
2. **Turn the knobs** — change the bottleneck size, the noise level, the sparsity/KL weight, the
   window length — and narrate *why* the result changes. That mechanism-level reasoning is exactly what
   an interview tests.
3. Exact numbers vary slightly by random seed — focus on the **trends and intuitions**, which are what
   generalize to your own problems.
