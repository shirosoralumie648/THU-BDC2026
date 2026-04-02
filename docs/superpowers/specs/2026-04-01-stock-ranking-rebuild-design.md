# Stock Ranking Rebuild Design

**Date:** 2026-04-01  
**Status:** Draft approved in-terminal, pending final user review  
**Scope:** THU-BDC2026 full internal rebuild for stock ranking, while preserving contest delivery compatibility

## Goal

Rebuild the internal ranking system so it can model cross-stock structure, intraday information, market regimes, and portfolio constraints more explicitly, without changing the external delivery contract.

The rebuilt system must continue to support:

- `train.sh`
- `test.sh`
- `output/result.csv`
- current Docker packaging and contest submission flow

The target is not only higher peak score, but better stability under rolling market regimes, lower concentration risk, and cleaner long-term maintainability.

## Non-Goals

This redesign does not attempt to:

- change the contest output schema
- require a new serving system outside local scripts and Docker
- depend on unavailable L2 order book data
- build a general-purpose quant research platform in one step

The redesign should improve the current repository, not replace it with an unrelated framework.

## Current State Assessment

The repository already contains a stronger baseline than a plain daily ranking model.

Existing strengths:

- cross-sectional rank features and industry-relative z-score features
- cross-stock attention with industry and correlation priors
- industry virtual tokens
- structured intraday factor aggregation
- multi-task volatility prediction
- mixed ranking loss with ListNet, Pairwise RankNet, and LambdaNDCG
- rolling validation and strategy reselection
- factor snapshotting and dataset lineage support

Current gaps that materially limit the system:

- graph structure is implicit in attention masks, not a first-class graph encoder
- intraday information is mostly engineered aggregation, not learned sequence encoding
- regime adaptation is gated, but not true router-plus-experts
- validation is rolling, but not fully purged with explicit embargo for `T+5`
- portfolio construction is still simple top-k selection with `equal` or `softmax`
- no turnover-aware optimization
- no ensemble orchestration
- no explicit feature importance or regime diagnostics loop

## Chosen Direction

The selected direction is `Modular Internal Rebuild with Compatibility Shell`.

Alternatives considered:

- `Incremental patching on top of the current StockTransformer`
  This is lower risk in the short term, but the current training script and model are already dense. Adding GNNs, learned intraday encoding, MoE routing, and portfolio constraints directly into the current monolith would make the system harder to change and debug.
- `Full platform rewrite`
  This would be the cleanest architecture, but it is too broad for the current contest-oriented repository. It would spend too much effort on framework concerns instead of ranking quality.

The chosen direction keeps the current entry points and artifacts, but moves the internal implementation to a layered, replaceable architecture.

## Compatibility Requirements

The following contracts are hard requirements:

- `train.sh` must still train the current best model flow
- `test.sh` must still generate the final submission file
- `output/result.csv` must remain the final output artifact
- existing Docker build and local validation scripts must continue to work
- legacy `train.csv` and `test.csv` workflows must remain available

Compatibility will be preserved by keeping `code/src/train.py` and `code/src/predict.py` as orchestration entry points, while moving most logic into new internal modules.

## Target Architecture

The rebuilt system is split into six logical layers and eight concrete module groups.

### Logical Layers

1. `Data Layer`
   Normalizes daily, intraday, industry, and macro inputs into reproducible canonical tables.

2. `Feature Layer`
   Produces daily, relative, intraday, and risk-normalized features.

3. `Relation Layer`
   Builds graph structures and graph-ready metadata for same-day stock interaction.

4. `Model Layer`
   Encodes daily sequences, intraday sequences, graph structure, and regime context into ranking representations.

5. `Strategy Layer`
   Converts stock scores into a constrained portfolio of at most five names.

6. `Experiment Layer`
   Handles validation, ensemble execution, metric selection, diagnostics, and experiment reproducibility.

### Concrete Module Groups

- `code/src/features/`
- `code/src/graph/`
- `code/src/models/`
- `code/src/objectives/`
- `code/src/portfolio/`
- `code/src/experiments/`
- `code/src/train.py`
- `code/src/predict.py`

The first six contain reusable internals. The last two remain compatibility wrappers.

## Module Boundaries

### `features`

Responsibilities:

- build daily factor tables
- build relative features
- build intraday structured features
- build risk-normalized features
- assemble wide training and inference tables

Core submodules:

- `daily_features.py`
- `relative_features.py`
- `intraday_features.py`
- `risk_features.py`
- `feature_assembler.py`

### `graph`

Responsibilities:

- build industry graph
- build rolling correlation graph
- optionally build holding-based graph later
- export graph tensors and metadata for training and inference

Core submodules:

- `industry_graph.py`
- `correlation_graph.py`
- `graph_builder.py`

### `models`

Responsibilities:

- daily sequence backbone
- intraday sequence encoder
- graph relation encoder
- regime router
- multi-head prediction

Core submodules:

- `daily_encoder.py`
- `intraday_encoder.py`
- `relation_encoder.py`
- `regime_router.py`
- `rank_model.py`

### `objectives`

Responsibilities:

- top-k aware ranking loss
- auxiliary volatility loss
- auxiliary drawdown loss
- adversarial training utilities
- regularization and calibration logic

Core submodules:

- `ranking_loss.py`
- `aux_losses.py`
- `adversarial.py`
- `target_transforms.py`

### `portfolio`

Responsibilities:

- select final names
- enforce industry diversification
- compute constrained weights
- apply turnover-aware adjustments

Core submodules:

- `candidate_selector.py`
- `constraints.py`
- `weighting.py`
- `policy.py`

### `experiments`

Responsibilities:

- purged walk-forward split generation
- multi-seed run orchestration
- ensemble score aggregation
- metric calculation and model selection
- diagnostics export

Core submodules:

- `splits.py`
- `runner.py`
- `ensemble.py`
- `metrics.py`
- `diagnostics.py`

## End-to-End Data Flow

The redesigned pipeline runs through five stages.

### Stage 1: Daily Base Build

Inputs:

- `stock_data.csv`
- `train.csv`
- `test.csv`
- optional structured dataset manifests

Outputs:

- canonical daily market table keyed by `instrument_id + trade_date`

Purpose:

- unify training and inference input shape
- isolate path and schema handling from model code

### Stage 2: Intraday Build

Inputs:

- minute bars when available

Outputs:

- structured intraday factors
- optional intraday tensor cache for learned sequence encoding

Purpose:

- support both a no-minute fallback path and a learned intraday path

### Stage 3: Feature Assembly

Inputs:

- daily factors
- relative features
- intraday structured features
- risk-normalized features
- static stock context

Outputs:

- wide feature table compatible with the current ranking dataset builder

### Stage 4: Graph Assembly

Inputs:

- industry mapping
- rolling historical returns
- optional future holding-based relationships

Outputs:

- `edge_index`
- `edge_weight`
- graph metadata

The graph layer must be optional. If graph inputs are incomplete, the model must fall back to attention-only operation rather than fail outright.

### Stage 5: Dataset Build

Each training sample represents one trading day and contains:

- `daily_sequence`
- `intraday_sequence` or `intraday_embedding`
- `graph`
- `static_context`
- `future_return_label`
- `future_volatility_label`
- `future_max_drawdown_label`

This is a stricter and more expressive sample contract than the current single wide-sequence input.

## Feature Design

The rebuilt feature system should keep the current alpha library, but add four feature groups that map directly to the requested improvements.

### 1. Relative Features

Add:

- stock return minus industry mean return
- stock return minus industry median return
- stock momentum versus industry leader momentum
- industry breadth and leader-follow strength proxies

Purpose:

- make relative strength explicit instead of relying only on hidden interaction layers

### 2. Risk-Normalized Features

Add:

- `return_5 / rolling_vol_20`
- `amihud / realized_vol`
- `intraday_tail_ret / daily_vol`
- other normalized momentum and liquidity features

Purpose:

- preserve feature meaning across low-vol and high-vol regimes

### 3. Intraday Structured Features

Add:

- intraday skewness
- downside return ratio
- jump ratio
- maximum minute shock
- Amihud-style impact proxies
- tail volume and amount concentration

Constraint:

- no fake order-book imbalance signals
- if only OHLCV minute bars exist, derive only proxy variables that are justified by the data

### 4. Learned Intraday Embeddings

When minute bars are available, support:

- CNN front-end for local microstructure shapes
- GRU sequence encoder for intraday progression
- one embedding per stock per day

When minute bars are missing, fall back to structured intraday factors only.

## Model Design

The new model should be modular instead of monolithic.

### Daily Encoder

Recommended default:

- retain a Transformer-based daily sequence encoder for the 60-day history window

Reason:

- the repository already uses this paradigm successfully
- replacing everything at once would add unnecessary migration risk

### Intraday Encoder

Two supported modes:

- `aggregate_only`
- `cnn_gru_encoder`

`aggregate_only` is the safe fallback and should remain available for low-resource runs or missing minute data.

### Relation Encoder

Support two interchangeable modes:

- `attention_relation`
- `gnn_relation`

The recommended first graph model is lightweight GraphSAGE or GAT over the daily candidate graph.

Reason:

- enough to express spillover and follow-on effects
- much cheaper and easier to stabilize than a large graph stack

### Regime Router

Replace single market gating with a router over three experts:

- `trend expert`
- `mean reversion expert`
- `defensive high-vol expert`

Inputs:

- market turnover
- realized volatility
- limit-up breadth
- market median return
- optional macro variables such as term spread when available

Outputs:

- expert weights
- regime diagnostics

The current market gating logic may be reused inside one expert, but should no longer be the top-level regime mechanism.

### Multi-Head Outputs

The rank model should emit:

- `rank_score`
- `future_volatility`
- `future_max_drawdown`
- optional `confidence_score`

The confidence score is not a contest artifact by itself. It exists to support portfolio weighting and diagnostics.

## Training Objective

The objective should be split into three parts.

### Primary Objective

- Top-k weighted LambdaRank or LambdaNDCG
- stronger penalty for misordering the real top names

Design rule:

- misplacing a true top-5 name to rank 20 should be penalized much more heavily than errors deep in the tail

### Auxiliary Objectives

- future 5-day volatility prediction
- future 5-day maximum drawdown prediction

Purpose:

- teach the shared representation to separate unstable winners from robust winners

### Robustness Objectives

- label transform from raw future return to excess-return-aware cross-sectional target
- Huber-style treatment for extreme events
- FGSM adversarial training on inputs

The primary task remains ranking, not pure classification. A quantile-aware target transform is acceptable, but the model must still output a ranking score for the full stock universe.

## Validation and Experiment Design

### Purged Walk-Forward Validation

Replace the current rolling validation with:

- walk-forward folds
- explicit `5 trading day embargo`
- no leakage between training and validation windows

This matters because the holding horizon is `T+5`.

### Multi-Seed Ensemble

Run:

- at least 5 seeds

Training outputs:

- per-model validation metrics
- per-model inference scores
- ensembled scores by average or robust mean

Inference ranking should be based on the ensemble score rather than the raw score of one model.

### Selection Metrics

Model and strategy selection should use more than one metric.

Required metrics:

- top-k risk-adjusted return
- top-5 excess return
- top-5 hit quality
- industry concentration penalty
- turnover-adjusted return

RankIC remains useful, but it should not be the only decision metric for a portfolio limited to five names.

### Diagnostics

Export:

- factor contribution summaries
- expert usage frequency
- regime distribution over time
- selected industry distribution
- turnover statistics

SHAP support is useful but not required for phase 1. A lighter internal attribution report is acceptable initially.

## Portfolio Policy Design

The current inference logic selects the top names and applies `equal` or `softmax` weighting. This is not sufficient for a highly concentrated portfolio.

The new portfolio logic should be two-stage.

### Stage 1: Candidate Ranking

The model produces:

- raw rank score
- risk estimate
- optional confidence score

### Stage 2: Constrained Portfolio Policy

The policy selects at most five names under these default rules:

- at least 3 distinct first-level industries when feasible
- at most 2 names from the same industry
- weights based on confidence and inverse risk
- optional turnover penalty against the previous portfolio when such state is available

Fallback:

- if constraints cannot be satisfied because of missing industry metadata or insufficient candidate diversity, fall back to the current compatible policy

### Weighting Rule

Recommended default:

- risk-parity-like weighting using confidence-adjusted inverse volatility

Not recommended as the default:

- full Kelly-style sizing

Reason:

- full Kelly sizing is too unstable for noisy contest predictions
- a simplified confidence-over-risk weighting is easier to control

## Phased Delivery Plan

### Phase 1: Stability and Policy Upgrade

Deliver:

- purged walk-forward validation
- portfolio policy with industry constraints
- ensemble orchestration
- richer intraday structured features
- risk-normalized features

Expected impact:

- fastest stability improvement for contest scoring

### Phase 2: Representation Upgrade

Deliver:

- CNN-GRU intraday encoder
- graph relation encoder
- drawdown auxiliary head
- FGSM adversarial training

Expected impact:

- better upside and more robust cross-regime generalization

### Phase 3: Regime and Explainability Upgrade

Deliver:

- mixture-of-experts regime router
- richer attribution and diagnostics
- optional holding graph support

Expected impact:

- better long-term maintainability and further score ceiling

## Migration Strategy

The migration should not be a single cutover.

Recommended sequence:

1. extract reusable internals from the current scripts without changing CLI behavior
2. introduce new modules behind compatibility wrappers
3. keep existing model path as a fallback baseline
4. add configuration switches to compare legacy and rebuilt paths
5. switch the default only after parity checks pass

This reduces the chance of a large rewrite breaking contest submission behavior.

## Testing Strategy

The rebuilt system should be covered at four levels.

### Unit Tests

- feature transforms
- graph construction
- loss functions
- portfolio constraints
- turnover penalty logic

### Integration Tests

- end-to-end train data build
- end-to-end predict data build
- model forward with and without intraday tensors
- graph and no-graph fallbacks

### Regression Tests

- `train.sh` still produces expected artifacts
- `test.sh` still writes `output/result.csv`
- Docker flow remains valid

### Evaluation Tests

- purged validation splitter produces non-overlapping folds
- portfolio policy respects industry limits when metadata exists
- ensemble output is reproducible for fixed seeds

## Risks and Controls

### Risk: Overbuilding the model before stabilizing validation

Control:

- phase delivery
- make purged validation and portfolio policy first-class before MoE

### Risk: Minute-level encoder adds complexity without stable data availability

Control:

- keep `aggregate_only` mode permanently supported

### Risk: Graph encoder overfits to noisy relationships

Control:

- start with simple industry and rolling-correlation graphs
- keep attention-only fallback

### Risk: Portfolio constraints reduce upside in strong single-industry trends

Control:

- make constraints soft by default where possible
- allow fallback and ablation flags

## Recommended First Implementation Plan

The first implementation plan derived from this design should focus on:

1. extracting modular packages without changing script entry points
2. implementing purged walk-forward validation
3. implementing portfolio policy with industry diversification and confidence-over-risk weighting
4. adding richer intraday structured features and risk-normalized features
5. adding multi-seed ensemble orchestration

This is the highest-return path before introducing the more expensive graph and expert-routing upgrades.
