# Color Palette Preference Modeling (CS329H)

End-to-end Python reimplementation of the MTurk color-palette preference pipeline from O’Donovan et al. (2011), extended with interpretable feature engineering and Bradley–Terry–style preference learning. The project compares a rating-based LASSO regression baseline to a pairwise Bradley–Terry logistic model on a five-color palette dataset.

## Repository Structure

- `analyze_palette_preferences.py` – Main experiment script. Implements:
  - Loading the MTurk palette dataset from `data/mturkData.mat`.
  - Color-space conversions (RGB, Lab, HSV, CHSV) and palette-level feature engineering.
  - LASSO regression on user-normalized ratings.
  - Bradley–Terry style logistic regression on synthetic pairwise preferences.
  - Single-seed analysis (plots, feature importance, error analysis) and multi-seed experiments with a paired t-test.
- `data/` – Original MATLAB code and datasets from “Color Compatibility From Large Datasets” (O’Donovan et al., ACM TOG 2011).
  - `mturkData.mat` – MTurk palette dataset used in this project (~10,743 palettes).
  - `data/hueProbsRGB.mat`, `data/kulerX.mat` – Auxiliary hue statistics used to construct hue-probability and CHSV features.
  - `weights.csv` – Feature weights from the original work, used for optional feature pruning.
  - `run.m`, `createFeaturesFromData.m`, `circstat/`, `glmnet_matlab/`, etc. – Original MATLAB reference implementation (not needed to run the Python pipeline).
  - See `data/README.txt` for license details and original documentation.
- `introduction_background.txt`, `methods_section.txt`, `results_discussion.txt`, `conclusion_future_work.txt` – Draft paper sections describing motivation, methods, results, and conclusions.
- Generated figures (created by running the pipeline):
  - `targets_hist.png`, `user_normalized_targets_hist.png`, `mean_rating_hist.png`
  - `lasso_true_vs_pred_test.png`, `lasso_calibration_test.png`
  - `lasso_feature_importance_top.png`, `bt_feature_importance_top.png`
  - `bt_most_certain_pair_*.png`, `bt_worst_mistake_*.png`

## Environment Setup and Dependencies

The analysis is implemented in Python and relies only on widely used scientific-computing libraries.

### Python version

- Tested with **Python 3.11**.
- Python **3.9+** is recommended.

### Recommended setup (virtual environment)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # on macOS/Linux
# On Windows (PowerShell): .venv\Scripts\Activate.ps1
```

### Required Python packages

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

These cover:

- `numpy`, `pandas` – array and table operations.
- `scipy` – `loadmat`, cubic splines, circular statistics utilities.
- `scikit-learn` – LASSO regression, logistic regression, scaling, and train/test splits.
- `matplotlib` – plotting and saving all figures (configured for non-interactive “Agg” backend inside the script).

No GPU or special system libraries are required beyond a standard C/Fortran toolchain for scientific Python wheels (typically provided by your Python distribution).

## Data and External Resources

All data needed to reproduce the core experiments are already included under `data/`:

- **Primary dataset**
  - `data/mturkData.mat` – MTurk five-color palette dataset with:
    - Palette IDs (`ids`) and names (`names`).
    - Mean ratings (`targets`).
    - User-normalized ratings (`userNormalizedTargets`) – the main prediction target.
    - Five RGB colors per palette (`data`), normalized to [0, 1].
- **Auxiliary hue statistics**
  - `data/data/hueProbsRGB.mat` – empirical hue distributions.
  - `data/data/kulerX.mat` – hue remapping grid.
  - Used to construct hue-probability features and CHSV (circular HSV) representations.
- **Original feature weights**
  - `data/weights.csv` – feature weights from the original work, used by the Python pipeline to select top Mturk-weighted features when available.

The `data/` directory is taken from:

> Peter O’Donovan, Aseem Agarwala, and Aaron Hertzmann. **Color Compatibility From Large Datasets.** ACM Transactions on Graphics (Proc. SIGGRAPH), 2011.

Please see `data/README.txt` for:

- License information (Creative Commons BY-NC-SA).
- Original documentation of the MATLAB code and datasets.

No additional downloads are required to run the experiments as long as `data/` is present.

## Step-by-Step: Reproducing the Results

The main entry point is `analyze_palette_preferences.py`, which orchestrates the full pipeline: data loading, feature construction, model training, evaluation, plotting, and multi-seed experiments.

### 1. Clone or open the project

Ensure you have this repository on your machine and a working directory at the project root (containing `analyze_palette_preferences.py` and the `data/` folder).

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### 4. Verify that data files are present

Check that the following files exist:

- `data/mturkData.mat`
- `data/data/hueProbsRGB.mat`
- `data/data/kulerX.mat`
- `data/weights.csv`

These are already included in the repository; if any are missing, restore them from your project backup or the original dataset distribution described in `data/README.txt`.

### 5. Run the full experiment pipeline

From the project root:

```bash
python3 analyze_palette_preferences.py
```

What this does:

1. **Prepare data**
   - Loads `mturkData.mat` into a Pandas DataFrame.
   - Prints basic statistics and writes histogram plots:
     - `targets_hist.png`
     - `user_normalized_targets_hist.png`
     - `mean_rating_hist.png`
2. **Build palette features**
   - Converts RGB to Lab, HSV, and CHSV color spaces.
   - Computes palette-level descriptors (per-color coordinates, sorted coordinates, adjacent differences, basic statistics, plane-fitting features).
   - Incorporates hue-probability features using `hueProbsRGB.mat` and `kulerX.mat`.
   - Optionally prunes to top Mturk-weighted features using `weights.csv` if available.
3. **Train and evaluate models (single seed)**
   - Trains a LASSO regression model to predict `userNormalizedTargets`.
   - Trains a Bradley–Terry logistic regression model on synthetic pairwise comparisons derived from the rating data.
   - Evaluates both models on a shared set of synthetic test pairs, computing:
     - Test RMSE for LASSO.
     - Pairwise accuracy and ROC AUC for both models.
   - Generates diagnostic plots:
     - `lasso_true_vs_pred_test.png` – true vs. predicted ratings.
     - `lasso_calibration_test.png` – calibration curve by rating bin.
     - `lasso_feature_importance_top.png`, `bt_feature_importance_top.png` – top feature importances.
   - Saves visualizations of selected palette pairs:
     - `bt_most_certain_pair_*.png` – high-confidence correct cases.
     - `bt_worst_mistake_*.png` – high-confidence errors.
4. **Run multi-seed experiments**
   - Repeats training and evaluation across several random seeds.
   - Aggregates metrics (mean ± std) for:
     - LASSO test RMSE.
     - BT and LASSO pairwise accuracies.
   - Runs a paired t-test (via `scipy.stats.ttest_rel`) on per-seed pairwise accuracies to compare BT vs. LASSO.

The console output will include all key scalar metrics and summary statistics; the PNG files in the project root serve as visual counterparts to the plots described in the paper-style writeup.

### 6. Interpreting and matching the reported numbers

Running `python3 analyze_palette_preferences.py` should yield metrics that closely match those described in:

- `results_discussion.txt`
- `conclusion_future_work.txt`

For example, you should see:

- LASSO test RMSE on `userNormalizedTargets` around 0.22–0.23.
- Pairwise accuracies above 0.85 and ROC AUC around 0.94 for both BT and LASSO on shared test pairs.

Minor differences (in the third decimal place) may arise from library versions or platform details, but the overall pattern should replicate.

## Expected Runtime and Computational Requirements

The dataset size is modest (~10,743 palettes with engineered features), and both models are linear, so the pipeline is computationally light.

- **Hardware**
  - Intended for CPU-only execution; no GPU is required.
  - A typical modern laptop (e.g., 4-core CPU, ≥8 GB RAM) is sufficient.
- **Runtime**
  - Full pipeline (single-seed analysis + multi-seed experiments + plotting) typically completes within a few minutes on a modern laptop.
  - If you run on very low-power hardware, allow up to ~10–15 minutes.
- **Memory**
  - Peak memory usage is well below 4 GB for the default settings (10k+ palettes, tens of thousands of synthetic pairs, and a few hundred engineered features).

Because the code uses the non-interactive Matplotlib “Agg” backend and writes plots directly to PNGs, it can be run on headless servers or remote machines without a display.

## Extending or Modifying the Experiments

You can customize the experiments by editing `analyze_palette_preferences.py`, for example:

- Changing the number of synthetic training/test pairs for the BT model.
- Adjusting the `delta` threshold that controls how different ratings must be to create a training/test pair.
- Modifying the set of seeds or the feature selection strategy.

Be sure to keep the `data/` directory intact and respect the original dataset license when sharing derived work.

## Acknowledgments

This project builds on the datasets and MATLAB code released by:

> Peter O’Donovan, Aseem Agarwala, and Aaron Hertzmann. **Color Compatibility From Large Datasets.** ACM Transactions on Graphics (Proc. SIGGRAPH), 2011.

Please consult `data/README.txt` and the original paper for additional details and citation guidelines.

