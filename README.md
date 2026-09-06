# MLV-CRL: Multi-modal and Multi-View Causal Representation Learning

This repository contains the node-discovery implementation used in our study.
It replaces the CLIP-distance factor proposal stage of
[MLLM-CD](https://arxiv.org/abs/2509.17784) with shared content blocks learned
from partially observed view subsets, following the motivation of
[Multi-View Causal Representation Learning with Partial Observability](https://arxiv.org/abs/2311.04056).
An MLLM then names each learned block from contrastive sample pairs.

The released default is the selected full-block configuration from the paper. It
stops after factor naming and evaluates discovered nodes with NP, NR and NF.
It does not reproduce the ablation study or the later studies.

## Pipeline

```text
raw image/text samples
  -> deterministic image and text views
  -> frozen CLIP feature cache
  -> MultiView block learning
  -> block-specific contrastive pair selection
  -> Gemini factor naming
  -> node-only graph
  -> human-reviewed GT mapping
  -> NP / NR / NF
```

Phase A learns the shared content blocks. Phase B selects sample pairs that are
far apart in one block and close in the remaining blocks, then asks Gemini to
name the dominant factor. No edge is added to the released node-only graph.

## Selected configuration

The only experiment override distributed here is
[`configs/run4_selected.py`](configs/run4_selected.py):

| Setting | Value |
|---|---|
| Content selector | variable |
| Objective | InfoNCE |
| Training similarity | negative squared L2 distance |
| Validity gate | off |
| Decomposition | off |
| Rank normalization | off |
| Block deduplication | off |
| Naming similarity | L2 distance |
| Contrastive pairs per block | 10 |
| Other-block coefficient | `1 / (M - 1)` |
| Quantification mode | MultiView |
| Execution scope | nodes only |



## Environment

CUDA 12.1 and Python 3.10 were used for the reported experiments.

```bash
conda env create -f environment.yml
conda activate mllm-cd-extended
```

The first CLIP run downloads the pinned OpenAI CLIP checkpoint. Graphviz and
causal-learn remain in the environment because the retained MLLM-CD utilities
also contain FCI and counterfactual-refinement code.

## Gemini configuration

Factor naming requires a Gemini API key. Keep it in the environment; do not put
it in source files, config files or `.env` files committed to Git.

```bash
export API_KEY="YOUR_GEMINI_API_KEY"
export MODEL="gemini-3.1-flash-image"
```

The outer run seed is forwarded to Gemini where the API supports seeded
generation. Hosted model behavior can nevertheless change, so newly generated
factor names need not be byte-identical to the archived mappings.

## Data preparation

The repository contains the MAG and Lung CSV files and their source images.
Generated views, manifests, CLIP caches and model weights are intentionally not
tracked.

Create the views and manifests:

```bash
python -m mvext.augment --dataset both
```

Create the frozen CLIP feature caches:

```bash
python -m mvext.encoders --dataset both
```

These commands create the following ignored artifacts:

```text
mvext_data/MAG/manifest.csv
mvext_data/MAG/clip_cache.pt
mvext_data/Lung/manifest.csv
mvext_data/Lung/clip_cache.pt
mvext_data/<dataset>/views/...
```

The run4 text views use dataset-specific semantic keyword masks. The masks are
defined in `mvext/augment.py`; generated view files are derived data and can be
recreated from the included inputs.

## Running node discovery

The convenience entry point has independent preparation, training and naming
stages. Commands are run from the repository root.

### Phase A: learn the blocks

```bash
python run_selected.py --stage train --dataset both --seed 0
```

This writes:

```text
outputs/run4_selected/blocks/seed00/MAG/blocks.pt
outputs/run4_selected/blocks/seed00/Lung/blocks.pt
```

Because the selected configuration has the validity gate disabled, all defined
view subsets enter final training. `blocks.pt` is specific to the dataset and
outer seed.

### Phase B: name the factors

```bash
python run_selected.py --stage name --dataset both --seed 0
```

The important outputs are:

```text
outputs/MAG9_mv/run4_selected_MAG_seed00/iter1_factors.json
outputs/Lung4_mv/run4_selected_Lung_seed00/iter1_factors.json
outputs/<dataset-dir>/<run-id>/<model>/graph_iter1_<timestamp>.json
```

The graph contains the target and discovered factor nodes but no edges. Reusing
the same run ID also reuses `iter1_factors.json`; use a different output folder
if a fresh independent naming call is required.

### Run every stage

After exporting `API_KEY`, preparation, training and naming can be run together:

```bash
python run_selected.py --stage all --dataset both --seed 0
```

Use `CUDA_VISIBLE_DEVICES` to select a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python run_selected.py --stage all --dataset both --seed 0
```

To reproduce the five outer seeds, repeat Phase A and Phase B with `--seed 0`
through `--seed 4`.

## Reviewing factor-to-GT mappings

The release includes dataset-wide mappings accumulated during development:

```text
eval/mappers/MAG.json
eval/mappers/Lung.json
```

Known factor names reuse their existing mappings. New names are added to the
`unreviewed` bucket. Historical run lists were cleared in the release; syncing
a new result records its new run ID without changing the mapping semantics.

Synchronize a completed run:

```bash
python -m eval.gt_mapper sync MAG run4_selected_MAG_seed00
python -m eval.gt_mapper sync Lung run4_selected_Lung_seed00
```

Create compact review files:

```bash
python eval/mappers/mapping_helper.py init MAG
python eval/mappers/mapping_helper.py init Lung
```

Edit `MAG_unreviewed_helper.json` or `Lung_unreviewed_helper.json`. Each value is
one of the following:

```json
{
  "unreviewed factor": null,
  "spurious factor": [],
  "single-GT factor": [2],
  "multi-GT factor": [3, 2]
}
```

- `null`: not reviewed yet
- `[]`: map to null
- one code: map to one GT node
- multiple codes: the first is the primary mapping used by NP/NR/NF; the rest
  are recorded as `compound.secondary`

Code tables:

| Dataset | Codes |
|---|---|
| Lung | `1=lesion`, `2=age`, `3=gender`, `4=smoking` |
| MAG | `1=defects`, `2=color`, `3=size`, `4=aroma`, `5=taste`, `6=nutrition`, `7=recmd`, `8=juiciness` |

Validate and apply the review:

```bash
python eval/mappers/mapping_helper.py check MAG
python eval/mappers/mapping_helper.py apply MAG

python eval/mappers/mapping_helper.py check Lung
python eval/mappers/mapping_helper.py apply Lung
```

## Node evaluation

After all new names have been reviewed:

```bash
python -m eval.node_metrics MAG run4_selected_MAG_seed00
python -m eval.node_metrics Lung run4_selected_Lung_seed00
```

The evaluator reports NP, NR and NF together with predicted and matched node
counts.

Compound secondary mappings are diagnostic annotations and are not added to the
matched set.

## Scope of retained MLLM-CD utilities

`utils.py`, `causal_graph_utils.py` and `counterfactual_utils.py` retain the
original MLLM-CD-style annotation, FCI and counterfactual-refinement utilities.
They are kept to preserve the extension path toward full causal graph discovery.
Those stages are outside the evaluated scope of this release; the selected
configuration and `run_selected.py` stop after node discovery.

## Checks

Run the dependency-free release contract check after installing the environment:

```bash
python tests/test_release.py
```

It verifies the selected configuration, mapper schemas, cleared provenance,
node metric formula and raw image references.

## References

- Jin Li et al. *Revealing Multimodal Causality with Large Language Models.*
  NeurIPS 2025. [arXiv:2509.17784](https://arxiv.org/abs/2509.17784)
- Dingling Yao et al. *Multi-View Causal Representation Learning with Partial
  Observability.* ICLR 2024.
  [arXiv:2311.04056](https://arxiv.org/abs/2311.04056)
