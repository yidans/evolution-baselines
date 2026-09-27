# Local link prediction v2

This is a test against the strongest classic local heuristic, not a test against CN alone. Locked comparators are CN, RA and AA. Qualification keeps the controller's row-wise strongest comparator, requires 2% relative AP improvement (95% lower confidence bound), minimum win rate 0.5, and two of three public families. The 2% threshold was fixed before measuring candidates: RA was near AP 0.98 on the prior GrQc diagnostic, making 5% relative improvement infeasible there. No controller comparison or acceptance code changes.

## Public graphs and reproducibility

Eight quality splits each use dolphins, the complete 8-core of ca-HepTh (285 nodes, 2237 edges), and the complete 22-core of email-Eu-core (394 nodes, 10397 edges). Core order was selected from topology alone before AP measurements: the smallest k whose complete k-core has between 80 and 400 nodes. Convert directed input to simple undirected, remove self-loops, keep every node/edge of that core, then use the unchanged sender sampler. Do not select components or splits by score. `data/transformations.json` records sizes; `data/sources.json` records downloads. Public seeds are fixed from v1, with no performance rejection sampling and no candidate truncation.

`python3 download_graphs.py` fetches missing files and prepares edge lists; `--offline` only uses checked-in raw files. Then `python3 build_pack.py --workers 3` builds a fresh pack. A generated pack is not resampled in place. The sender's evaluator is imported unchanged for all new splits. Three separately seeded 400-node synthetic runtime envelopes remain runtime-only and do not contribute to AP qualification. Hidden registry, configs and all 16 saved hidden splits are copied byte-for-byte from v1; they are never regenerated.

## Sources and license status

- ca-HepTh: [SNAP source](https://snap.stanford.edu/data/ca-HepTh.html), arXiv High Energy Physics Theory collaboration, 1993–April 2003. Citation: Jure Leskovec, Jon Kleinberg, Christos Faloutsos, *Graph Evolution: Densification and Shrinking Diameters*, ACM TKDD, 2007.
- email-Eu-core: [SNAP source](https://snap.stanford.edu/data/email-Eu-core.html), anonymized email network from a European research institution. Citation: Hao Yin, Austin R. Benson, Jure Leskovec, David F. Gleich, *Local Higher-order Graph Clustering*, KDD, 2017; also Leskovec et al., 2007.
- Raw URLs: https://snap.stanford.edu/data/ca-HepTh.txt.gz and https://snap.stanford.edu/data/email-Eu-core.txt.gz.
- Those authoritative dataset pages do not state an explicit redistribution license. License status is recorded as **not specified**, not inferred from the SNAP software license. Source attribution and downloaded originals are preserved for this research pack.
- Dolphins and private graphs retain v1's sender provenance. `operator_sources/baselines.py` is the unmodified sender file. Only the selected CN/RA/AA function definitions plus their unchanged common helper are exposed in each public CLI implementation; the full operator file is not projected to model workspaces.

## Interfaces and metrics

Required candidate files are `solution.py` exposing `score_candidates(adj, candidates)` and `method_manifest.json` specifying a JSON CLI. Adapter filename is unrestricted. The candidate sees observed edges and candidate pairs; evaluation labels/metadata are stripped by the existing controller projection. AP groups ties exactly as sklearn; AUC is tie-aware. Hits@L uses stable candidate-index tie breaking in the pack; the sender's default NumPy argsort may break ties differently. This difference is retained from v1 and is not an AP difference.

Validation: `python3 scripts/validate_local_link_prediction_v2_pack.py` from repository root validates actual controller schemas and all 43 instances against all three baselines. `baseline_validation.json` is controller-private because it includes hidden results. Public baseline profiling exposes only public rows. No hidden measurement is used by RSI selection: evaluations run development-only, with the champion's full protocol reserved for lineage completion.
