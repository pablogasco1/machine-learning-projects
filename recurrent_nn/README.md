# Recurrent Neural Networks — A Hands-On Course

A progressive, heavily-commented series of Jupyter notebooks that teaches **Recurrent Neural
Networks (RNNs)** from first principles up to a realistic, consulting-grade forecasting project.
Everything runs **offline on CPU** using synthetic data — no downloads, no GPU required.

Built with **PyTorch**. Each notebook is self-contained but they are designed to be read in order.

## The path

| # | Notebook | What you'll learn | Key idea |
|---|----------|-------------------|----------|
| 01 | [`01_rnn_fundamentals.ipynb`](01_rnn_fundamentals.ipynb) | Sequential data, the hidden state as *memory*, the recurrence equation, "unrolling"/flow through time, the **5 RNN shapes**, and a **from-scratch NumPy RNN cell**. | *An RNN is one small rule applied over and over, carrying a memory forward.* |
| 02 | [`02_first_rnn_pytorch.ipynb`](02_first_rnn_pytorch.ipynb) | Tensor shapes `(batch, time, features)`, `nn.RNNCell` (manual loop) vs `nn.RNN`, the training loop, time-based splits, naive baselines, recursive multi-step forecasting. | *Your first trained RNN — predict the next value of a signal.* |
| 03 | [`03_lstm_gru.ipynb`](03_lstm_gru.ipynb) | The **vanishing-gradient** problem, **LSTM** (cell state + forget/input/output gates), **GRU** (reset/update gates), and a head-to-head proving LSTM/GRU beat vanilla RNNs on long memory. | *Gates create a memory "highway" that doesn't forget.* |
| 04 | [`04_many_to_one_classification.ipynb`](04_many_to_one_classification.ipynb) | **Many-to-one** sequence classification on multivariate vibration sensors (predictive maintenance): `Dataset`/`DataLoader`, an LSTM classifier, confusion-matrix evaluation. | *Read a sequence → emit one label (fault vs. healthy).* |
| 05 | [`05_one_to_many_generation.ipynb`](05_one_to_many_generation.ipynb) | **One-to-many** autoregressive generation: a character-level name generator, embeddings, and **temperature** sampling. | *Feed the model its own output to generate sequences.* |
| 06 | [`06_energy_sensors_forecasting.ipynb`](06_energy_sensors_forecasting.ipynb) | **Capstone:** high-frequency, multivariate **wind-farm sensor forecasting**. Sliding windows, a Direct multi-output GRU vs. an **Encoder–Decoder (seq2seq)** GRU with teacher forcing, baselines, error-vs-horizon, and business framing. | *Many-to-many multi-step forecasting for the energy industry.* |
| 07 | [`07_timeseries_features_gbdt.ipynb`](07_timeseries_features_gbdt.ipynb) | **The other paradigm:** don't feed raw time to a net — **engineer features → LightGBM**. Retail demand forecasting: leakage-safe lags/rolling/calendar/promo features, a global GBDT beating baselines, feature importance, recursive vs. direct multi-step, quantile intervals, and the **RNN-embedding → GBDT hybrid**. | *When trees on engineered features beat (or complement) an RNN.* |
| 08 | [`08_multivariate_finance_forecasting.ipynb`](08_multivariate_finance_forecasting.ipynb) | **Exogenous drivers:** forecast a **gold-miners ETF** from a basket (gold, S&P 500, WTI, dollar, 10Y/30Y yields, VIX, inflation) with **both** a multivariate LSTM and LightGBM. Finance discipline: returns-not-prices, no-look-ahead alignment, *explaining vs predicting*, directional accuracy, and a cost-aware **backtest**. | *Feeding many related series to predict one — done honestly.* |

## How to run

From this folder, launch Jupyter and open the notebooks in order:

```bash
jupyter lab      # or: jupyter notebook
```

Or run one end-to-end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_rnn_fundamentals.ipynb
```

All notebooks were executed and verified to run top-to-bottom on CPU.

## Requirements

Already covered by the repo's `requirements.txt`:

```
numpy · pandas · matplotlib · seaborn · scikit-learn · torch · lightgbm
```

> Notebooks 01–06 use **PyTorch**; notebook 07 uses **LightGBM** (with an optional mention of
> `tsfresh` for automated feature extraction, which isn't required to run anything).

## Suggested study approach

1. **Read the markdown first**, then run each cell and confirm the output matches the explanation.
2. **Experiment** — change `hidden_size`, sequence lengths, `temperature`, the forecast horizon,
   or swap `nn.RNN` ↔ `nn.LSTM` ↔ `nn.GRU` (a one-word change) and watch what happens.
3. Exact numbers vary slightly by random seed — focus on the **trends and intuitions**, which are
   what generalize to your own problems.
