# NP-AIRT implementation skeleton

> Historical AIRT design note. The current filtered benchmark-holdout work uses
> the standalone ANP in `organized/training/anp_crossattn_blocks.py`, documented
> in [`methods-guide.md`](methods-guide.md). This file is retained to explain
> the earlier IRT-anchored architecture. Do not treat its \(C\subseteq T\)
> objective or its feature contract as the current trainer.

Status: implemented experimental skeleton; not final competition evidence.
The canonical symbols are defined in [`notation.md`](notation.md).

## Model

For one subject \(s\), the acquired context and prediction targets are

\[
C=\{(x_i,y_{si})\}_{i=1}^{N_C},\qquad
T=\{(x_j,y_{sj})\}_{j=1}^{N_T}.
\]

During episodic training, \(C\subseteq T\). At inference, only
\((x_C,y_C,x_T)\) are available; \(y_T\) is hidden.

The amortized item network supplies the 2PL parameters

\[
(a_j,b_j)=f_\phi(x_j),\qquad a_j=\operatorname{softplus}(\tilde a_j)+\epsilon.
\]

The latent path encodes each pair into \(s_i\), aggregates it into \(s_C\),
and parameterizes \(q(z\mid s_C)\). During training, the same encoder produces
\(q(z\mid s_T)\). The ANP deterministic path encodes \(r_i\) and uses the
target input \(x_j\) to obtain \(r_j^*\) by cross-attention.

The IRT-anchored decoder is

\[
\theta_s=g_\psi(z),\qquad
\ell_{sj}=a_j(\theta_s-b_j)+\delta_\omega(x_j,z,r_j^*),
\qquad p(y_{sj}=1)=\sigma(\ell_{sj}).
\]

For the latent-only NP variant, omit \(r_j^*\). The residual \(\delta\) is an
additional predictive term; it is not renamed as item difficulty.

## Code-shaped pseudocode

```python
def training_loss(x_C, y_C, x_T, y_T, n_z_samples=L):
    x_C_encoded = encode_x(x_C)
    x_T_encoded = encode_x(x_T)

    q_z_C = q_z(x_C_encoded, y_C)  # q(z | s_C)
    q_z_T = q_z(x_T_encoded, y_T)  # q(z | s_T), training only
    z = q_z_T.rsample((L,))

    r_star = deterministic_path(x_C_encoded, y_C, x_T_encoded)
    logits_T = decoder(z, x_T_encoded, r_star)

    nll = binary_cross_entropy_with_logits(logits_T, y_T)
    kl = KL(q_z_T, q_z_C)
    return nll + kl / N_T


def predict(x_C, y_C, x_T, n_z_samples=L):
    q_z_C = q_z(encode_x(x_C), y_C)
    z = q_z_C.rsample((L,))
    r_star = deterministic_path(encode_x(x_C), y_C, encode_x(x_T))
    p_y_T_given_z = sigmoid(decoder(z, encode_x(x_T), r_star))
    return mean(p_y_T_given_z, axis=0), variance(p_y_T_given_z, axis=0)
```

At budget zero, \(C=\varnothing\) and the latent path uses its explicit prior.
Evaluation averages a fixed number of Monte Carlo samples; it never reports one
uncontrolled draw. The variance is computed in probability space.

## Episodic data contract

The prepared training task is `{"task_id", "x", "y"}`. An evaluation task is

```python
{
    "x_C_pool": ...,  # fixed acquisition order
    "y_C_pool": ...,  # revealed only up to budget B
    "x_T": ...,       # untouched evaluation items
    "y_T": ...,       # scorer only
}
```

For each training episode, randomly choose \(N_C\), select indices for \(C\),
then construct \(T\) so it contains those context indices plus additional
targets. Context aggregation is permutation-invariant; target outputs are
permutation-equivariant.

## Training-only trace supervision

Response traces are privileged offline information, not submission-time model
inputs. A trace teacher may add an auxiliary or distillation loss, but the
student path must remain

\[
(x_C,y_C,x_T)\longrightarrow p(y_T=1\mid x_C,y_C,x_T).
\]

Mask trace losses where traces are unavailable. Never derive a feature from a
validation or test target trace. A release test must reproduce predictions
with the trace directory absent.

## Implementation ladder

1. Frozen amortized 1PL base (difficulty only, unit discrimination).
2. Frozen amortized 2PL base.
3. Regularized online MAP update of \(\theta_s\).
4. Deterministic CNP-AIRT.
5. Latent NP-AIRT with \(q(z\mid s_C)\).
6. ANP-AIRT with target-specific \(r_j^*\).
7. Stacked self-attention in the \(s\) and \(r\) paths.
8. Relation-aware attention only after the simpler controls are sound.

The encoder-contribution ladder in `modal_cnp_ladder.py` follows rungs 1, 2,
3, and 4, then removes the CNP correction \(\Delta\theta(C)\) at inference and
retrains a correction-free CNP item head. The point is to test whether the set
encoder contributes anything beyond an item head plus MAP ability.
