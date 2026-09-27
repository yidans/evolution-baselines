"""Link-prediction baseline suite: classical heuristics, best-of-N sampling, OpenEvolve-style search.

Every arm is evaluated by the same code path (:mod:`.evaluate`) on the same v2 pack
and judged by the controller's own ``aggregate_development_qualification_v2``.
Hidden held-out graphs are only touched through :func:`.evaluate.CandidateEvaluator.heldout`,
which the search arms never call.
"""
