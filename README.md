# Causal Debt Dataset Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A synthetic data generator for causal inference on debt forgiveness scenarios. This repository provides a framework to create realistic financial datasets with known causal structures for evaluating causal inference methods with continuous treatments.

## Overview

This repository contains code for generating synthetic financial data that simulates a debt forgiveness scenario with continuous treatment effects, where:
- Treatment: Percentage of debt written down (0-100%)
- Outcome: Binary indicator of debt repayment (0 = default, 1 = repayment)
- Features: Realistic financial indicators (credit history, debt profiles, account information)

The data generation process is designed to incorporate key challenges in causal inference:
- High dimensionality (can generate hundreds of features)
- Systematic confounding bias
- Positivity assumption violations
- Heterogeneous treatment effects

## Repository Structure

```
causal-debt-datagen/
├── src/
│   ├── data_generation.py     # Main data generation script
│   └── ...
├── config.py                  # Configuration parameters
├── requirements.txt           # Package dependencies
└── README.md                  # This documentation
```

## Data Generation Process

The synthetic data contains three types of variables:
1. **Confounders**: Variables that influence both treatment assignment and outcome
2. **Treatment-only variables**: Variables that affect only treatment assignment
3. **Outcome-only variables**: Variables that affect only the repayment outcome

### Variable Definitions

| Variable | Description |
|----------|-------------|
| X₁ᵢ | Years since default |
| X₂ᵢ | Default debt amount |
| X₃ᵢ | Number of loans |
| X₄ᵢ | External debt |
| X₅ᵢ | Number of cards |
| X₆ᵢ | Loss given default |
| X₇ᵢ | Number of refinances |
| X₈ᵢ | Customer history length |
| X₉ᵢ | Number of accounts |
| X₁₀ᵢ | Months since first payment |

All features are standardized before applying the treatment and outcome formulas.

### Treatment Assignment Formula

The treatment value (percentage of debt loss) for each individual is generated through:

```
Tᵢ = clip(100 · (1/(1+e^(-θᵀXᵢ))) + εᵢ, 0, 100)
```

where the linear predictor θᵀXᵢ is defined as:

```
θᵀXᵢ = 0.5X₁ᵢ + 0.4log(1+X₂ᵢ) + 0.3X₃ᵢ + 0.3log(1+X₄ᵢ) + 
       0.2X₅ᵢ + 0.3X₆ᵢ + 0.2X₇ᵢ + 0.1X₁ᵢlog(1+X₂ᵢ) + 0.1X₃ᵢ²
```

and εᵢ follows a truncated normal distribution:

```
εᵢ ~ TN(0, σ²=25, a=0, b=100)
```

### Outcome Generation Formula

The outcome generation involves two stages: probability computation and sampling with monotonicity constraints.

For each individual, the base probability of repayment given treatment t is:

```
P(Yᵢ = 1|T = t, Xᵢ) = {
    0                    if t = 0
    1                    if t = 100
    (t/100)^exp(ηᵢ)      otherwise
}
```

where the individual coefficient ηᵢ is computed through:

```
ηᵢ = 0.6X₁ᵢ + 0.5log(1+X₂ᵢ) + 0.5X₃ᵢ + 0.4log(1+X₄ᵢ) + 0.3X₅ᵢ - 
     0.4X₈ᵢ - 0.3X₉ᵢ - 0.2X₁₀ᵢ + 0.1X₁ᵢlog(1+X₂ᵢ) + 0.1X₃ᵢ²
```

These probabilities are clipped to [0,1] range and sampled from a Bernoulli distribution:

```
p̃ᵢ = clip(P(Yᵢ = 1|T = t, Xᵢ), 0, 1)
Yᵢ ~ Bernoulli(p̃ᵢ)
```

The final dose-response curves incorporate monotonicity constraints where observed outcome at treatment t_obs influences potential outcomes:

```
P(Yᵢ = 1|T = s, Xᵢ) = {
    0    if Yᵢ = 0 and s ≤ t_obs
    1    if Yᵢ = 1 and s ≥ t_obs
    p̃ᵢ   otherwise
}
```

## Usage

```python
from src.data_generation import DataGeneration

# Initialize the data generator
data_gen = DataGeneration()

# Generate synthetic financial data (10,000 samples)
df = data_gen.generate_random_financial_data(n_samples=10000)

# Fit scalers on the data
data_gen.fit_scalers(df)

# Generate treatment values
treatment = data_gen.generate_treatment(df)
df['treatment'] = treatment

# Calculate outcome probability and binary outcome
probs, outcome = data_gen.calculate_outcome_probability(df, treatment)
df['outcome_probability'] = probs
df['outcome'] = outcome

# Optionally add synthetic features for more complexity
df_with_extra = add_synthetic_features(df, n_redundant=100, n_noise=300)
```

## Citation

If you use this dataset generator in your research or applications, please cite:

```bibtex
@software{causal_debt_datagen,
  author = {Javier Moral Hernández},
  title = {Causal Debt Dataset Generator},
  url = {https://github.com/javiermoralh/causal-debt-datagen.git},
  year = {2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.