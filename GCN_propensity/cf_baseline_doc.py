\section{Methodology}

This section describes the construction of the implicit feedback dataset, the
weighting schemes used to encode trading intensity, the collaborative filtering
model configuration, and the evaluation metrics employed to assess predictive
performance. The objective is to develop a daily recommendation model that
identifies which fixed-income instruments (CUSIPs) each counterparty is most
likely to trade in the near future, based solely on historical trading activity.

\subsection{Problem Setting}

Let $\mathcal{U}$ denote the set of counterparties (users) and $\mathcal{I}$
the set of traded instruments (items). The raw dataset consists of triplets
\[
    (\mathrm{u}, \mathrm{i}, t, a_t),
\]
where $\mathrm{u} \in \mathcal{U}$, $\mathrm{i} \in \mathcal{I}$,
$t$ is the trade date (with daily granularity), and $a_t$ is an optional trade
attribute such as notional amount. The task is to estimate, for each day $t$
and each pair $(\mathrm{u}, \mathrm{i})$, a relevance score
$\hat{r}_{\mathrm{u},\mathrm{i}}(t)$ quantifying the probability that the user
will trade instrument $\mathrm{i}$ on day $t$.

We treat this as an \emph{implicit-feedback recommendation problem} because the
observed trades do not provide explicit ratings but instead represent positive
binary or weighted signals. All unobserved $(\mathrm{u},\mathrm{i})$ pairs are
treated as implicit negatives during the training process.

\subsection{Data Aggregation and Weighting}

Because the data may contain multiple trades per
day, and because trade notionals vary widely, we collapse trades to a daily
user--item matrix with one interaction weight per triplet $(\mathrm{u},\mathrm{i},t)$.

For each counterparty–CUSIP–day triple we compute:
\[
    c_{\mathrm{u},\mathrm{i},t} = \text{count of trades on day } t,
\]
\[
    n_{\mathrm{u},\mathrm{i},t} = \text{total notional traded on day } t.
\]

Depending on the configuration, the daily interaction strength
$w_{\mathrm{u},\mathrm{i},t}$ is computed using one of the following weighting
schemes:

\begin{itemize}
    \item \textbf{Binary:}
          \[
              w_{\mathrm{u},\mathrm{i},t} = 1.
          \]

    \item \textbf{Frequency-based:}
          \[
              w_{\mathrm{u},\mathrm{i},t} = \log(1 + c_{\mathrm{u},\mathrm{i},t}).
          \]

    \item \textbf{Notional-based:}
          \[
              w_{\mathrm{u},\mathrm{i},t} = \sqrt{n_{\mathrm{u},\mathrm{i},t}}.
          \]

    \item \textbf{Hybrid (frequency and notional):}
          \[
              w_{\mathrm{u},\mathrm{i},t}
              = \log(1 + c_{\mathrm{u},\mathrm{i},t}) \, \sqrt{n_{\mathrm{u},\mathrm{i},t}}.
          \]

    \item \textbf{Time-decay:} recent trades receive more weight:
          \[
              w_{\mathrm{u},\mathrm{i},t}
              = \log(1 + c_{\mathrm{u},\mathrm{i},t}) \cdot
                \exp\!\left(-\frac{\Delta t}{\tau}\right),
          \]
          where $\Delta t$ is the number of days prior to the current date and
          $\tau$ is a decay horizon (typically $\tau \in [20,40]$).
\end{itemize}

The total interaction strength over the training window is:
\[
    W_{\mathrm{u},\mathrm{i}} = \sum_{t \in \mathcal{T}_{\mathrm{train}}}
    w_{\mathrm{u},\mathrm{i},t}.
\]

Finally, to ensure numerical stability and prevent users with unusually high
trading volume from dominating the model, the weights are normalized
\emph{per user}:
\[
    \tilde{W}_{\mathrm{u},\mathrm{i}}
    = \frac{W_{\mathrm{u},\mathrm{i}}}
           {\max_{j \in \mathcal{I}} W_{\mathrm{u},j} + \epsilon},
\]
where $\epsilon$ is a small constant (e.g.\ $10^{-9}$).

\subsection{Model Configuration}

We employ the Alternating Least Squares (ALS) algorithm for implicit feedback
\cite{hu2008implicit}, implemented in the \texttt{implicit} Python package.
ALS is particularly suited for large sparse interaction matrices and scales
well to tens of thousands of users and items.

The model approximates the implicit interaction matrix
$\tilde{W} \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ as
\[
    \tilde{W} \approx UV^\top,
\]
where $U \in \mathbb{R}^{|\mathcal{U}| \times d}$ and
$V \in \mathbb{R}^{|\mathcal{I}| \times d}$ are latent factor matrices, and $d$
is the embedding dimension.

The confidence model used by ALS is:
\[
    C_{\mathrm{u},\mathrm{i}} = 1 + \alpha \tilde{W}_{\mathrm{u},\mathrm{i}},
\]
where $\alpha$ controls the relative importance of confidence versus preference
(e.g.\ $\alpha = 15$).

Key hyperparameters:

\begin{itemize}
    \item Number of latent factors: $d = 64$
    \item Regularization parameter: $\lambda = 0.1$
    \item Number of ALS iterations: $20$
    \item Confidence scaling parameter: $\alpha = 15$
\end{itemize}

The model optimizes:
\[
\min_{U,V} \sum_{\mathrm{u},\mathrm{i}}
    C_{\mathrm{u},\mathrm{i}}
    \left( P_{\mathrm{u},\mathrm{i}} - U_{\mathrm{u}}^\top V_{\mathrm{i}} \right)^2
    + \lambda \left( \|U\|^2 + \|V\|^2 \right),
\]
where $P_{\mathrm{u},\mathrm{i}} = 1$ if $(\mathrm{u},\mathrm{i})$ is observed and $0$
otherwise.

Scores for recommendation are computed via:
\[
    \hat{r}_{\mathrm{u},\mathrm{i}} = U_{\mathrm{u}}^\top V_{\mathrm{i}}.
\]

\subsection{Train--Test Splitting}

Because the problem is time-dependent, we use a strictly chronological split.
Let $T_{\max}$ denote the most recent date in the dataset. The test period is
defined as the last $K$ days:
\[
    \mathcal{T}_{\mathrm{test}}
    = [T_{\max} - (K-1), \, T_{\max}],
\]
with all prior data used for training:
\[
    \mathcal{T}_{\mathrm{train}} = \{t : t < T_{\max} - (K-1)\}.
\]

A typical value is $K = 30$ days.

\subsection{Evaluation Metrics}

Performance is evaluated as a top-$k$ ranking problem. For each user active in
the test set, the model produces a ranked list of candidate instruments based on
$\hat{r}_{\mathrm{u},\mathrm{i}}$. Let $\mathcal{I}^{+}_{\mathrm{u}}$ be the set of test-period
instruments actually traded by user $\mathrm{u}$.

\paragraph{Recall@k.}
Recall@k measures the proportion of relevant items retrieved among the top-$k$
recommendations:
\[
    \mathrm{Recall@k}(\mathrm{u})
    =
    \frac{
        |\mathcal{I}^{+}_{\mathrm{u}} \cap \mathrm{TopK}(\mathrm{u})|
     }{ |\mathcal{I}^{+}_{\mathrm{u}}| }.
\]
The final metric is averaged over all users with at least one test interaction.

\paragraph{Mean Average Precision (optional).}
For completeness, the model may also be evaluated with:
\[
    \mathrm{MAP@k}
    =
    \frac{1}{|\mathcal{U}_{\mathrm{test}}|}
    \sum_{\mathrm{u}}
    \frac{1}{|\mathcal{I}^{+}_{\mathrm{u}}|}
    \sum_{j=1}^{k} \mathrm{Prec@j}(\mathrm{u}) \cdot \mathbb{1}\{ \text{item at rank $j$ is relevant} \}.
\]

\paragraph{Ranking Interpretation.}
These metrics focus specifically on the ability of the model to place relevant
CUSIPs within the top portion of the ranking list, which is aligned with the
operational goal of generating a daily shortlist of high-propensity trade ideas
for each counterparty.

\subsection{Summary}

The resulting baseline system is a fully implicit collaborative-filtering model
that incorporates trading intensity through multiple weighting schemes, respects
temporal causality through chronological splitting, and provides daily
rank-ordered trade recommendations. This framework establishes a robust and
interpretable foundation upon which more sophisticated graph-based or
time-dependent models (e.g.\ LightGCN, GNN-based causal convolution) can be
developed.

