# Canonical notation

This project uses the notation of the Neural Process (NP), Attentive Neural
Process (ANP), and amortized item-response theory literature. New code,
equations, plots, logs, and reports must use this vocabulary.

## Sets and observations

| Symbol | Code | Meaning |
|---|---|---|
| \(C\) | `C` | Acquired context set for one subject--benchmark task |
| \(T\) | `T` | Target set used for prediction; current filtered NP training uses \(C\cap T=\varnothing\) |
| \(x_i\) | `x` | Information available for item \(i\) before acquiring its outcome |
| \(y_{si}\) | `y` | Binary outcome for subject \(s\) on item \(i\) |
| \(x_C,y_C\) | `x_C`, `y_C` | Context inputs and acquired outcomes |
| \(x_T,y_T\) | `x_T`, `y_T` | Target inputs and outcomes; `y_T` is unavailable at inference |
| \(N_C,N_T\) | `n_C`, `n_T` | Context and target set sizes |

An evaluation task stores the full ordered acquisition pool as `x_C_pool` and
`y_C_pool`. At budget \(B\), the context contains the **accepted** labels from
the stream up to that point, not necessarily its first \(B\) candidates. Under
random-prefix acquisition those are the same. The untouched evaluation pool is
`x_T`, with `y_T` used only to score predictions.

The original CNP reconstruction objective uses \(C\subseteq T\). We keep that
notation only when describing that historical paper-faithful control; it is not
the active filtered NP training rule.

## NP and ANP paths

| Symbol | Code | Meaning |
|---|---|---|
| \(r_i\) | `r_i` | Deterministic representation of context pair \((x_i,y_i)\) |
| \(r_*\) | `r_star` | Target-specific deterministic representation obtained by attention |
| \(s_i\) | `s_i` | Latent-path representation of context pair \((x_i,y_i)\) |
| \(s_C\) | `s_C` | Permutation-invariant aggregation of latent representations for \(C\) |
| \(s_T\) | `s_T` | Equivalent aggregation over \(T\), used only in the training posterior |
| \(z\) | `z` | Global stochastic latent variable |
| \(q(z\mid s_C)\) | `q_z_C` | Context-conditioned latent distribution used at inference |
| \(q(z\mid s_T)\) | `q_z_T` | Target-conditioned latent distribution used to train the latent path |
| \(L\) | `n_z_samples` | Number of Monte Carlo samples of \(z\) |

The italic subscript \(s\) in \(y_{si}\) identifies a subject; the symbol
\(s_i\) without a subject subscript is the ANP paper's latent-path encoding.

For ANP, the decoder is

\[
p(y_*\mid x_*,r_*,z).
\]

For a latent-only NP variant, \(r_*\) is omitted. The code never calls an
encoded input `embedding`, `feature`, or `representation` when the paper symbol
is more precise: the raw model input remains `x`, its encoded form is
`x_encoded`, and the two NP paths produce `r` and `s`.

## IRT and amortized IRT

| Symbol | Code | Meaning |
|---|---|---|
| \(\theta_s\) | `theta` | Ability of subject \(s\) |
| \(a_i>0\) | `a` | Item discrimination |
| \(b_i\) | `b` | Item difficulty |
| \(c_i\in[0,1]\) | `c` | Item guessing floor in 3PL |
| \(\phi\) | `phi` | Parameters of an amortized item network |

The model family uses

\[
\begin{aligned}
\text{1PL: }&p(y_{si}=1)=\sigma(\theta_s-b_i),\\
\text{2PL: }&p(y_{si}=1)=\sigma\!\left(a_i(\theta_s-b_i)\right),\\
\text{3PL: }&p(y_{si}=1)=c_i+(1-c_i)\sigma\!\left(a_i(\theta_s-b_i)\right).
\end{aligned}
\]

Amortized IRT predicts item parameters from item input rather than storing a
free parameter for every item:

\[
(a_i,b_i,c_i)=f_\phi(x_i),
\]

with the appropriate positivity and probability constraints. In NP-AIRT,
`theta_from_z` maps the task latent \(z\) to \(\theta_s\), and the Bernoulli
decoder retains the IRT logit as its psychometric anchor.

## Reserved naming rules

- `C` always means observed/acquired context; `T` always means targets.
- `y` means an observed binary outcome, never a probability. Use `p_y_T` for
  predicted target probabilities.
- Use `mu_z_C`, `sigma_z_C`, and analogous names for distribution parameters.
- Use `a`, `b`, and `c` only for discrimination, difficulty, and guessing.
- Use `theta` only for subject ability.
- Use `r` for the deterministic path and `s` for the latent encoder path.
- Use `z` for the sampled global latent variable, not for ability itself.
- Use `R` for a train-only reference set in an EBR acquisition calculation. It
  is never the hidden evaluation set.
