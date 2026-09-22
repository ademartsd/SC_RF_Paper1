# MH-MCMC — Sedimentary Environment

Bayesian inversion of receiver functions (RFs) using a Metropolis-Hastings Markov Chain Monte Carlo (MH-MCMC) approach.

**Output:** `MATRIX_res_{station}.csv`  
**Author:** Ademar Fernández, September 2026  
**Contact:** cfernandez@seoe.sc.edu

---

## Part 1: Inversion Code

The inversion code is:

`MH_MCMC_SED.py`

### Running the inversion

Activate the Conda environment:

```bash
conda activate prs
```

Then run:

```bash
python3 MH_MCMC_SED.py
```

### Input files

The inversion requires:

1. RF stream: `stream_Z56A.h5`
2. RF stack: `Z56A_trace_mean.txt`
3. RF standard deviation: `Z56A_trace_std.txt`

### Output

Once the inversion is running, the following file will be created inside the `output` directory:

```text
./output/MATRIX_res_Z56A.csv
```

> **Full inversion:** Comment out line 48 to perform the full run.  
> The full inversion takes approximately **two months** to complete.

---

## Part 2: Plotting Script

The plotting script is:

`PLOTS_SED.py`

### Running the plotting script

Activate the Conda environment:

```bash
conda activate prs
```

Then run:

```bash
python3 PLOTS_SED.py
```

### Input files

The plotting script requires:

1. Inversion results: `./output/MATRIX_res_Z56A.csv`
2. RF stream: `stream_Z56A.h5`
3. RF stack: `Z56A_trace_mean.txt`
4. RF standard deviation: `Z56A_trace_std.txt`

### Output

The script creates the following directory:

```text
./figs/
```

All generated figures are stored inside this directory.

### Plotting options

- **Line 5 — `number_to_plot`:** Controls the number of models to plot.
- **Line 9:** Controls whether all RF traces are used.

Using all traces produces the realistic/full plot and can take approximately **24 hours** to complete. The fast-view option takes only a few minutes.