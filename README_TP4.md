# Machine Learning – Practical Session 4

**Student:** Sofia Belouahar  
**Group:** AIA2 G1

Overfitting, data leakage and model generalization.

## Goal of the TP

The goal is to predict the final CGPA (`final_cgpa`) of 1,200 students from a survey, and to see how a bad evaluation (data leakage) or a too complex model (overfitting) gives results that look good but do not generalize. All the models are compared with the same chronological split (first 80% of the survey for training, last 20% for testing).

## Files

- `TP4_Solutions.ipynb`: the notebook with the code, the outputs, the plots and the written answers (Parts A to D and the final discussion)
- `hybrid_student_performance_1200.csv`: the dataset
- `README.md`: this file

## How to run

Install the libraries (scikit-learn 1.3 or newer is needed for `TargetEncoder`):

```bash
python -m pip install numpy pandas matplotlib scikit-learn ipykernel
```

Then open `TP4_Solutions.ipynb` in VS Code, choose a Python kernel with these libraries and click **Run All**. A fixed seed (`random_state=42`) is used everywhere so the results are reproducible.

## What I did in each part

### Part A: data inspection

- 1,200 rows and 36 columns, 13 columns have between 1.9% and 4.8% of missing values (35.3% of the rows have at least one).
- No duplicates in the file. The file is not sorted by date, so I sorted it by `timestamp` once and reused this order everywhere.
- I built `final_cgpa`, the encoded features, `cohort_code` and `effort_index`. The strongest correlations with the target are `effort_index` (0.38), `attendance_rate` (0.37) and `study_consistency` (0.32). The weakest are `sleep_hours`, `stress_level`, `submission_day` and `age` (close to 0).
- `performance_risk_level` is strongly linked to the CGPA (alone it gives R² = 0.40), so it cannot be used as a feature (target leakage).
- `cohort_code` has 42 categories and 12 of them have less than 10 students.
- No clear trend of `final_cgpa` over time.

### Part B: data leakage

Training and testing RMSE (the baseline predicts the training mean):

| Strategy | Training RMSE | Testing RMSE | Baseline testing RMSE |
|---|---:|---:|---:|
| B.1 Random split | 0.0000 | 1.5420 | 1.3447 |
| B.1 Chronological split | 0.0000 | 1.9585 | 1.2362 |
| B.2 Split before removing duplicates | 0.0000 | 1.5758 | 1.2767 |
| B.2 Remove duplicates before split | 0.0000 | 1.6525 | 1.2181 |
| B.3 Scaler fitted on all data | 1.0681 | 1.1496 | 1.2362 |
| B.3 Scaler fitted on train only (Pipeline) | 1.0655 | 1.1454 | 1.2362 |

- B.1: the full decision tree memorizes the training set and is worse than the baseline on the test set. The chronological split gives a worse score mostly because of `submission_day` (without it the two splits are close).
- B.2: with 15% of duplicated rows, 50 of the 276 test rows have a copy in the training set, so the score looks better (1.5758) than the honest one (1.6525).
- B.3: the difference is very small here (1.1496 vs 1.1454), but the scaler and the imputer must still be learned on the training data only, that is why everything is inside a `Pipeline`.

### Part C: overfitting

- C.1: with the decision tree, the best testing RMSE is at depth 3 (1.0677). Overfitting starts at depth 5 (gap 0.2803) and without limit the training RMSE is 0.0000 but the testing RMSE is 1.9585.
- C.2: with polynomial features of `effort_index`, the CV RMSE stays almost the same from degree 1 to 9 (1.1774 to 1.1883) and gets worse at degree 15 (1.2941). The effect is much smaller than for the tree because there is only one feature and the polynomial is a smooth curve.

### Part D: improving generalization

- D.1: for `cohort_code`, grouping the rare categories is the best treatment (validation RMSE 1.2211), then feature removal (1.2408), then target encoding (1.2605). The variable is only weakly informative.
- D.2: removing `submission_day` improves the three models, so I removed it.
- D.3: with degree 5, Ridge (alpha = 10) and Lasso (alpha = 0.01) improve a little the validation RMSE (1.1854 without penalty, 1.1782 with Ridge and 1.1771 with Lasso). Lasso puts 4 of the 5 coefficients exactly to zero.
- D.4: final comparison on the test set (used only once):

| Model | Training RMSE | Testing RMSE | Testing R² |
|---|---:|---:|---:|
| Baseline (training mean) | 1.2685 | 1.2362 | -0.0029 |
| Linear Regression | 1.0717 | 1.0853 | 0.2270 |
| Lasso (alpha = 0.1) | 1.1064 | 1.0721 | 0.2457 |
| Decision Tree (max_depth = 3) | 1.1021 | 1.0677 | 0.2519 |
| Gradient Boosting (135 trees) | 0.9554 | 1.0593 | 0.2636 |

## Conclusion

- The Gradient Boosting with early stopping has the best test score, but the differences between the four models are small (RMSE between 1.0593 and 1.0853 on only 240 students), so I cannot say there is a clear winner.
- The best R² is only 0.26: the survey answers are coarse ranges and the target has only 4 values, so most of the variation of the CGPA is not explained.
- The techniques that reduced overfitting the most are limiting the depth of the tree, regularization (Lasso/Ridge) and early stopping.
- A high score is not always good: the full tree has an error of 0 on the training set but is worse than the baseline on the test set, and `performance_risk_level` alone gives a higher R² than the honest models because it is linked to the target.
