# MVCD: Multi-View Causal Discovery

This repository contains the implementation accompanying *Enhancing MLLM-Based
Causal Discovery with Multi-View Causal Representation Learning*. MVCD replaces
the CLIP-distance factor proposal stage of
[MLLM-CD](https://arxiv.org/abs/2509.17784) with factor blocks learned from
partially observed view subsets, following
[Multi-View Causal Representation Learning with Partial Observability](https://arxiv.org/abs/2311.04056).
An MLLM names and quantifies the learned factors, and FCI estimates a partial
ancestral graph (PAG).

The default is the paper configuration. Ablation runners and development
outputs are intentionally excluded.

## Pipeline

```text
raw image/text samples
  -> deterministic image views and corpus-frequency text views
  -> frozen CLIP feature cache
  -> MultiView factor-block learning
  -> block-specific contrastive pair selection
  -> Gemini factor naming
  -> Gemini per-sample factor quantification (-1/0/1)
  -> KCI-FCI PAG estimation with the observed target score
  -> human-reviewed factor-to-GT mapping
  -> NP / NR / NF / AP / AR / AF / ESHD
```

Factor naming and graph estimation are separate stages. GT mappings are not
used by FCI and may be reviewed after naming while graph estimation runs; they
are required only for evaluation.

## Paper configuration

[`configs/mvcd.py`](configs/mvcd.py) selects:

| Setting | Value |
|---|---|
| Text-view vocabulary | corpus term frequency, top 6 |
| Content selector | variable |
| Objective | InfoNCE |
| Training similarity | negative squared L2 distance |
| Validity gate | off |
| Decomposition / rank normalization / deduplication | off |
| Naming pair similarity | L2 distance |
| Contrastive pairs per block | 10 |
| Other-block coefficient | average, `1/(M-1)` |
| Factor naming mode | one name per MultiView block |
| FCI quantification | MLLM-CD ternary annotation |
| Graph scope | iteration 1 only |

The naming mode and FCI quantification are deliberately separate. Naming keeps
one factor per learned block without an LLM merge. Only after the names are
fixed does Gemini assign every sample a value in `{-1, 0, 1}` for FCI.

## Environment

CUDA 12.1 and Python 3.10 were used for the reported experiments.

```bash
conda env create -f environment.yml
conda activate mllm-cd-extended
```

The first encoder run downloads the pinned OpenAI CLIP checkpoint. Graph
estimation requires `causal-learn`; Graphviz/pydot support PAG rendering and
serialization. These dependencies are included in `environment.yml`.

## Gemini configuration and data disclosure

Naming and per-sample quantification require a Gemini API key:

```bash
export API_KEY="YOUR_GEMINI_API_KEY"
export MODEL="gemini-3.1-flash-image"
```

These stages send dataset text, images and observed target scores to the
configured external Gemini service. Confirm that this is permitted for your
data before running them. The outer seed is forwarded where the API supports
seeded generation, but hosted model updates can still prevent byte-identical
responses.

## View construction

For each dataset, MVCD uses three image views:

- mean-color image;
- Gaussian high-pass image;
- Otsu-thresholded silhouette.

For text, MAG uses `Review`; Lung concatenates `Demographics` and `History`.
The corpus is lowercased and tokenized into unigrams. English stop words and
terms occurring in fewer than two or more than 90% of documents are removed.
The six remaining terms with highest total corpus frequency are selected, with
alphabetical tie-breaking.

For each selected term, the other five selected terms are split into two
disjoint groups. Masking one group in each copy produces two complementary
views whose selected-vocabulary intersection retains the target term. This
gives 12 text views. Together with the three image views, each dataset has 15
views and nine two-view subsets: three image subsets and six text subsets. No
GT factor name or GT factor count enters this construction.

Generate views and their manifest:

```bash
python run_mvcd.py --stage prepare --dataset both --seed 0
```

Important generated files are:

```text
mvext_data/<dataset>/text_view_spec.json
mvext_data/<dataset>/manifest.csv
mvext_data/<dataset>/clip_cache.pt
mvext_data/<dataset>/views/...
```

`text_view_spec.json` records the ranking, selected terms and preprocessing
settings. These derived artifacts are ignored by Git and can be regenerated.

## Running MVCD

Commands below run from the repository root. Use `CUDA_VISIBLE_DEVICES` to
select a GPU, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python run_mvcd.py --stage train --dataset both --seed 0
```

### 1. Learn factor blocks

After preparing views, run:

```bash
python run_mvcd.py --stage train --dataset both --seed 0
```

This writes:

```text
outputs/mvcd/blocks/seed00/MAG/blocks.pt
outputs/mvcd/blocks/seed00/Lung/blocks.pt
```

The validity gate is disabled in the paper configuration, so all nine subsets
enter full-data training.

### 2. Name factors

```bash
python run_mvcd.py --stage name --dataset both --seed 0
```

Important outputs are:

```text
outputs/MAG9_mv/mvcd_MAG_seed00/iter1_factors.json
outputs/MAG9_mv/mvcd_MAG_seed00/top_pairs.json
outputs/Lung4_mv/mvcd_Lung_seed00/iter1_factors.json
outputs/Lung4_mv/mvcd_Lung_seed00/top_pairs.json
```

Incomplete factor-name/criteria responses are retried up to three additional
times. Reusing the same run ID reuses the completed factor file.

### 3. Quantify factors and estimate a PAG

```bash
python run_mvcd.py --stage graph --dataset both --seed 0
```

Gemini assigns `-1`, `0` or `1` to every named factor for every sample. The
observed `score` column is appended, and FCI is run with the KCI independence
test and `alpha=0.05`. Counterfactual iteration 2 is not part of this command.

```text
outputs/<dataset-dir>/mvcd_<dataset>_seed00/iter1_annotation_results.csv
outputs/<dataset-dir>/mvcd_<dataset>_seed00/fci_input_iter1.csv
outputs/<dataset-dir>/mvcd_<dataset>_seed00/graph_iter1.json
outputs/<dataset-dir>/mvcd_<dataset>_seed00/manifest.json
```

### Combined execution

Preparation, training, naming and graph estimation can run sequentially:

```bash
python run_mvcd.py --stage all --dataset both --seed 0
```

Manual mapping and evaluation are intentionally not included in `all`. Repeat
with `--seed 0` through `--seed 4` for the five reported outer seeds.

## Reviewing factor-to-GT mappings

The release includes dataset-wide mappings accumulated across the experiments:

```text
eval/mappers/MAG.json
eval/mappers/Lung.json
```

Known names reuse their reviewed mappings. New names enter `unreviewed` when a
run is synchronized:

```bash
python -m eval.gt_mapper sync MAG mvcd_MAG_seed00
python -m eval.gt_mapper sync Lung mvcd_Lung_seed00
```

Create compact review files:

```bash
python eval/mappers/mapping_helper.py init MAG
python eval/mappers/mapping_helper.py init Lung
```

Edit `MAG_unreviewed_helper.json` or `Lung_unreviewed_helper.json`:

```json
{
  "not reviewed": null,
  "spurious factor": [],
  "single-GT factor": [2],
  "multi-GT factor": [3, 2]
}
```

- `null`: not reviewed;
- `[]`: map to null;
- one code: primary GT mapping;
- multiple codes: first is primary, remaining codes are recorded as
  `compound.secondary` and do not alter the paper metrics.

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

## Evaluation

After graph generation and mapping review:

```bash
python run_mvcd.py --stage evaluate --dataset both --seed 0
```

The evaluator writes `metrics_iter1.json` in each run directory. NP/NR/NF
include the observed target node. Predicted and matched factor diagnostics
exclude the target. AP/AR/AF compare adjacency on the matched-node subgraph.
ESHD adds matched-subgraph SHD, GT edges incident to missing nodes, predicted
edges incident to spurious nodes, and one penalty per missing/spurious node.
Only the paper's GT-reference metrics are exposed; faithful and global variants
from development experiments are not included.

## Retained MLLM-CD code

`utils.py`, `causal_graph_utils.py` and `counterfactual_utils.py` retain the
original MLLM-CD annotation, FCI and counterfactual-refinement utilities. MVCD
uses the annotation and FCI portions. Counterfactual refinement remains
available for future extension but is outside the released paper pipeline.

## Checks

After installing the environment, run:

```bash
python tests/test_release.py
```

The check covers the paper configuration, deterministic text vocabulary/view
construction, mapper schemas, graph metrics and raw image references.

## References

- Jin Li et al. *Revealing Multimodal Causality with Large Language Models.*
  NeurIPS 2025. [arXiv:2509.17784](https://arxiv.org/abs/2509.17784)
- Dingling Yao et al. *Multi-View Causal Representation Learning with Partial
  Observability.* ICLR 2024.
  [arXiv:2311.04056](https://arxiv.org/abs/2311.04056)
