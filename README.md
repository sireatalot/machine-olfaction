Machine Olfaction Test Task
===========================
Option A: Model Improvement

Reproduction
------------
- Create and activate a virtual environment: `python -m venv env`
- Activate virtual environment: `source env/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run the analysis from the repo root: `python demo/dominant_balanced_tau.py`
- Inputs: `demo/Mixture.csv` and `demo/Intensities.csv` (provided).
- Outputs: `demo/mixture_intensities_with_dominance.csv` plus console diagnostics for the best fixed `tau`, adaptive `tau` stats, and mean top-weight summary.


By Ronaldo Tatang <rtatang@umich.edu>