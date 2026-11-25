"""Generate fake test-score data for the hierarchical-model exercise.

Model:
    theta_city   ~ N(mu, tau^2)    tau is a standard deviation    one mean score per city
    score_school ~ N(theta_city, sigma_school^2)    one score per school

Notation:
    n_city             number of cities (groups)
    n_school_per_city  number of schools in each city

Output file `fake_school_data.txt` has shape (n_school_per_city, 2 * n_city):
column 2k holds the scores of the schools in city k, column 2k + 1 their variances.
"""
import numpy as np

rng = np.random.default_rng(2026)

mu_true = 70.0  # population mean of the city means
tau_true = 10.0  # population standard deviation of the city means

n_city = 20
n_school_per_city = 10

# Per-city mean scores, shape (n_city,)
theta_true = rng.normal(loc=mu_true, scale=tau_true, size=n_city)

# Per-school measurement standard deviations, shape (n_city, n_school_per_city)
sigma_school = rng.uniform(4.0, 12.0, size=(n_city, n_school_per_city))

# Observed scores, shape (n_city, n_school_per_city)
scores = theta_true[:, None] + sigma_school * rng.normal(size=(n_city, n_school_per_city))

# Interleave (scores, variances) per city -> shape (n_school_per_city, 2 * n_city)
columns = []
for city_index in range(n_city):
    columns.append(scores[city_index])
    columns.append(sigma_school[city_index] ** 2)
table = np.column_stack(columns)

theta_string = ", ".join(f"{theta:.2f}" for theta in theta_true)
np.savetxt(
    "fake_school_data.txt",
    table,
    fmt="%.4f",
    header=(
        f"mu={mu_true}, tau={tau_true} (standard deviation), theta=[{theta_string}]\n"
        "columns: score_city1 variance_city1 score_city2 variance_city2 ..."
    ),
)
