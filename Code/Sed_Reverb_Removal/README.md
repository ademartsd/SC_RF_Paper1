# Sediment Reverberation Filter

Code for removing sediment-generated reverberations from receiver functions (RFs).

**Input:** Raw RF stream containing sediment reverberations  
**Output:** Curated RF stream  
**Author:** Dr. Shubham Agrawal  
**Contact:** SAGRAWAL@mailbox.sc.edu  
**Additional contact:** cfernandez@seoe.sc.edu

---

## Citation

If you use this code, please cite the following work together with the accompanying manuscript:

> Agrawal, S., Eakin, C.M., O’Donnell, J.P., 2023. Tracking crustal thickness at the sediment-inundated edge of the Gawler Craton, South Australia. *Tectonophysics* 862, 229938.

---

## Running the Code

The sediment reverberation removal code is:

`reverb_removal.py`

Activate the Conda environment:

```bash
conda activate prs
```

Then run:

```bash
python3 reverb_removal.py
```

### Input

The code requires the raw RF stream containing sediment reverberations:

```text
stream_Z56A_RAW_STILL_REVERB.h5
```

### Output

Once the code is running, the curated RF stream will be created inside the `output` directory:

```text
./output/stream_Z56A.h5
```

This curated RF stream is subsequently used as the RF input for the MH-MCMC inversion of station Z56A.