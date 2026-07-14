# Physics-Informed Neural Networks (PINNs) — Hands-On Series

A guided, notebook-by-notebook introduction to **PINNs**, built to go from *"I've never seen this"* to
*"I can explain a real energy business case on a whiteboard."* Every notebook is self-contained, runs on **CPU
in a couple of minutes**, and explains each new concept before using it.

> **Why this series exists.** Energy systems (heat in pipes, temperature of a battery, pressure in a reservoir,
> voltage in a circuit) are governed by *differential equations from physics*, but real assets give us only a
> *few noisy sensors*. PINNs are the technique that fuses the two: trustworthy physics + imperfect data.

## The idea in one paragraph

A PINN is an ordinary neural network whose output is trained to **satisfy a differential equation**. We use
automatic differentiation to compute the derivatives that appear in the equation (e.g. `d(NN)/dt`), form a
**residual** that is zero when the physics holds, and minimize it — optionally together with a **data term**
that matches measurements. No labelled solution is required; the physics supplies the supervision.

## The notebooks (do them in order)

| # | Notebook | Physics | New concepts | Energy relevance |
|---|----------|---------|--------------|------------------|
| 01 | `01_pinn_basics_exponential_decay.ipynb` | 1st-order ODE `dy/dt=-k y` | autograd, residual, collocation points, initial condition | RC circuit / cooling / decay |
| 02 | `02_pinn_damped_oscillator.ipynb` | 2nd-order ODE (spring/RLC) | 2nd derivatives, two ICs, loss weighting, **spectral bias** | machine vibration, resonance |
| 03 | `03_pinn_heat_equation_forward.ipynb` | 1D heat PDE `u_t=α u_xx` | PDEs (space+time), boundary conditions, field visualisation | thermal conduction, insulation |
| 04 | `04_pinn_inverse_parameter_discovery.ipynb` | heat PDE, **α unknown** | **inverse problems**, trainable physical parameters, sensor fusion | digital twin, condition monitoring |
| 05 | `05_pinn_business_district_heating.ipynb` | advection–diffusion + heat loss | outflow BC, normalization, classical baseline, **KPI in €** | district-heating operations + monitoring |
| 06 | `06_pinn_differentiable_surrogate_control.ipynb` | parametric steady pipe `θ(x; v)` | **control as an input**, freeze twin, **back-prop through it** to optimise | optimal pump-speed / design / MPC |
| 07 | `07_pinn_data_assimilation_virtual_sensors.ipynb` | heat PDE, **known** α | **data assimilation**, virtual sensing, denoising, recover unmeasured state | soft sensors, gap-filling |

**Skim path if short on time:** read 01 fully → run 03 → study 04 and 05 (these are the interview gold).
06 and 07 are the two "extra pattern" notebooks that flesh out the ✅ rows of the table below
(differentiable-surrogate/control and sensor-fusion/assimilation).

## When to actually use a PINN (the judgement table)

| Situation | PINN? | Why |
|---|---|---|
| Known equation, simple geometry, need speed | ❌ | Classical FD/FE/CFD solvers are faster & battle-tested |
| **Unknown parameter to infer from data** | ✅ | Inverse problems are the sweet spot (nb 04, 05) |
| **Fuse sparse/noisy sensors with physics** | ✅ | Data-efficient, physically consistent, interpretable (nb 04, 05, **07**) |
| Need a **differentiable surrogate** for optimisation/control | ✅ | One smooth model to back-propagate through (nb **06**) |
| Highly oscillatory / sharp shocks / long horizons | ⚠️ | Spectral bias — needs special tricks or may fail (nb 02) |
| No physics known, lots of data | ❌ | Use ordinary ML / time-series models |

## Running

Everything uses packages already in the repo's `requirements.txt` (`torch`, `numpy`, `matplotlib`, `nbformat`).

```bash
# from the repo root, with your venv active
pip install -r requirements.txt
jupyter lab            # then open the PINNS/ folder and run top-to-bottom
```

No GPU needed. The ODE notebooks (01–02) train in seconds; the PDE notebooks (03–07) take ~2–4 minutes each on
CPU (the `u_xx` term needs double automatic differentiation, which is the expensive part). All notebooks are
shipped **already executed**, so you can also just read them. Tip: run them **one at a time** — several heavy
notebooks training at once contend for CPU.

## The recurring lesson

PINNs shine when you have **trustworthy physics but imperfect data** — exactly the condition of a real energy
asset. Forward PINNs teach the mechanics; **inverse PINNs (parameter discovery from sparse sensors) are where
the business value lives.**
