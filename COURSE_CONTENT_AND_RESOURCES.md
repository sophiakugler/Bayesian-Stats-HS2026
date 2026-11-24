# Bayesian Statistic Methods and Data Analysis, HS2026

Instructors: Patrick Meyers, Uddipta Bhardwaj, Giada Badaracco

Course times, location: Wednesdays 9:45 - 12:30

# Resources

## Literature

Below are a list of common resources. I have bolded the ones that I think are good to learn the foundations of statistics from. 

- Practical Statistics for Astronomers, Wall, 2012 [ETH library](https://eth.swisscovery.slsp.ch/permalink/41SLSP_ETH/lshl64/alma99117170816205503). Short, with a focus on practical applications. Many of the examples and exercises are from astrophysics but are generally applicable, especially for the physical sciences, which often come a bit short in general statistics textbooks. Solutions and data sets are [available online](https://www.astro.ubc.ca/people/jvw/ASTROSTATS/pracstats_web_ed2.html).
- Statistics, data mining, and machine learning in astronomy, 2020 [ETH library](https://eth.swisscovery.ch/permalink/41SLSP_ETH/13pv5mr/alma990114457340205503). A commonly used book for astrophysics data anlaysis courses. 
- **Data Analysis: A Bayesian Tutorial, Sivia and Skilling, 2006** [ETH library](https://eth.swisscovery.ch/discovery/fulldisplay?docid=alma99121186169505503&context=L&vid=41SLSP_ETH:ETH&lang=en&search_scope=DiscoveryNetwork&adaptor=Local%20Search%20Engine&tab=discovery_network&query=any,contains,sivia%20and%20skilling&sortby=date_d&facet=frbrgroupid,include,9047206918619755727&mode=basic&offset=0). The first 4 chapters of this are the best intro to Bayesian statistics I have found.

- Bayesian Data Analysis, Gelman et al, 2013 [ETH library](https://eth.swisscovery.slsp.ch/permalink/41SLSP_ETH/lshl64/alma99117222397805503), [Link](http://www.stat.columbia.edu/~gelman/book/). The title says it all. This is a great *reference* but not the best to learn the basics from. We may use it for the model checking parts of the course, as those chapters are very nice. 
- **Statistical Rethinking, McElreath, 2020** [ETH library](https://eth.swisscovery.slsp.ch/permalink/41SLSP_ETH/lshl64/alma99117227648305503). A very nice introduction to Bayesian statistics. The book uses `R` and `stan` but implemtations in other languages are available. Unfortunately, not available online as a PDF.
- Information Theory, Inference, and Learning Algorithms, MacKay, 2003 [Link](http://www.inference.org.uk/itprnn/book.pdf). Heavy on the information theory but also covers inference methods nicely. The exercises come with solutions.
- Weighing the odds, a course in probability and statistics, Williams, 2001 [ETH library](https://eth.swisscovery.slsp.ch/permalink/41SLSP_ETH/lshl64/alma99117170967205503). A good introduction to probability theory and statistics with a high level of mathematical rigour.


## Examples of other courses

- Past years of this course [here](https://github.com/tilmantroester/bayesian_statistical_methods
). We will follow quite closely what has been done in the past.
- Vanderbilt University, Stephen Taylor [here](https://github.com/VanderbiltAstronomy/astr_8070_s25
). This course is maybe the closest to what we will be doing.
- University of Minnesota, Michael Coughlin [here](https://github.com/UMN-Big-Data-in-Astrophysics
). This has a focus on large datasets and really applying statistical methods.
- University of Oregon, Ben Farr [here](https://github.com/uo-phys/comp-phys-26
). More of a projec-based focus. We may pull some suggested project topics or datasets from here.


## Other useful resources

- Python bootcamp [here](https://github.com/fedhere/PyBOOT).
- Andrew Gelman blog [here](https://statmodeling.stat.columbia.edu) (statistics blog with focus on social sciences).
- Git (from Michael Coughlin) [cheatsheet](https://github.com/UMN-Big-Data-in-Astrophysics/ast8581_2025_Spring/blob/main/help/git/git-cheat-sheet.pdf), [workflow](https://github.com/UMN-Big-Data-in-Astrophysics/ast8581_2025_Spring/blob/main/help/git/git-transport-v1-1024x723.png), [instructions on syncing forks](https://github.com/UMN-Big-Data-in-Astrophysics/ast8581_2025_Spring/blob/main/help/git/sync_a_fork.md)

# Course outline

The course outline below is from the previous version of this course taught by Tilman Troester.
We may change things as we go, but this should serve as a roadmap for people for what to expect. 



week11_gaussian_processes.ipynb
week11_gps






week7_model_checking.pdf


## 1. Intro
Notebook: `lectures/week1_intro.ipynb`
- JupyterLab
- Debugging
- Version control
- Python packages
### Exercise
- Use JupyterLab
- Monte Carlo estimate of pi

## 2. Probabilities
Notebook: `lectures/week2_probabilities.ipynb`
- Different definitions of probability
- Set notation
- Outcomes, events
- Kolmogorov axioms
- Conditional probabilities and independence
- Bayes theorem
### Exercises
- Birthday problem
- Monty Hall problem

## 3. Random variables and probability distributions
Notebook: `lectures/week3_random_variables_and_probability_distributions.ipynb`
- Random variables
- Probability distributions: discrete and continuous
- PDF and CDF
- Change of variables
- Inverse transform sampling
- Expectation
- Mean, variance, moments
- Joint, conditional, and marginal distributions
- Common probability distributions
    - Uniform
    - Binomial, multinomial
    - Poisson
    - Gaussian
    - Chi-squared
    - Cauchy
    - Power law
    - Central limit theorem
### Exercise
- Inverse transform sampling
- Derive Poisson from binomial distribution
- Distribution of sum of Gaussian
- General sum of independent RVs
- Distribution of chi-squared distribution

## 4. Introduction to Bayesian statistics
Notebook: `lectures/week4_intro_to_bayes.ipynb`
- Bayes theorem
- Likelihood, prior, posterior
- Updating priors
- Prior and posterior predictive distributions
- Model comparison: evidences and Bayes ratio
- Bayesian line fitting
- MAP
- Posterior sampling
- Computing predictive distributions
### Exercises
- Fitting data
- Misspecified likelihood

## 5. Sampling from distributions 1
Notebook: `lectures/week5_sampling.ipynb`
- Monte Carlo estimates of integrals
- Rejection sampling
- Markov chain Monte Carlo
- Metropolis-Hastings
### Exercises
- Implement rejection sampling
- Implement Metropolis-Hastings in n-d
- Show that Metropolis-Hastings satisfies detailed balance

## 6. Sampling from distributions 2
Notebook: `lectures/week6_sampling_2.ipynb`
- Burn-in, convergence, and auto-correlation
- Slice sampling
- Nested sampling
- Application to model selection using Bayes' ratio on super novae data
### Exercises
- Implement nested or slice sampling
- Use emcee and dynesty
- Use dynesty to compare models

## 7. Model checking
Notebook: `lectures/week7_model_checking.ipynb`
- Chi-square goodness-of-fit
- Posterior predictive checks
- Model comparison:
    - DIC
    - WAIC
    - Cross-validation
### Exercises
- Implement chi-square and posterior predictive checks
- Use DIC, WAIC, and Bayes ratio for model comparison

## 8. Model Comparison and Linear Regression 
Notebook: `lectures/week_8_model_comparison_and_regression.pdf`
- Statistics and estimators
- Estimator bias and variance
- Statistics and their sampling distributions
    - Sample mean
    - Sample variance
    - Sample covariance
    - Correlation coefficient
- Correlation
    - Malmquist bias
- PCA
- Bootstrap
### Exercises
- Show that the sample variance estimator is unbiased
- Compute posterior on the correlation coefficient
- Check bootrap on case where exact sampling distribution is known

## 9. Fisher, Hamilton Monte Carlo, and JAX
Notebook: `lectures/week9_fisher_hmc_jax_numpyro.ipynb`
- Fisher information matrix
- Cramer-Rao bound
- Jeffreys prior
- JAX
- Hamiltonian Monte Carlo
### Exercises
- Use JAX to get Fisher information
- Experiment with HMC settings
- Use implementation of NUTS in tensorflow-probability

## 10. Simulation-based inference
Notebook: `lectures/simulation_based_inference.ipynb`
- Approximate Bayesian computation
- Neural density estimation
- Kullback-Leibler divergence
- Gaussian mixture models
- Loss functions and posteriors
- MLPs
- L_2, L_1, negative log likelihood loss
### Exercises
- Implement rejection ABC
- Show that the function that minimises the L_1 loss is the median
- Implement neural density estimation 

## 11. Gaussian processes & probabilistic programming
Notebook: `lectures/week11_gaussian_processes.html`
- Gaussian processes
- Kernel functions & Gaussian processes regression with tinygp
- Sampling GP parameters

## 12. Gaussian random fields & recap
Notebook: ``
- Gaussian random fields
- Power spectrum
Notebook: `lectures/recap.ipynb`
- Recap of the course
