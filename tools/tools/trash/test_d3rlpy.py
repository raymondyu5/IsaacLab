import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')

# prepare algorithm
cql = d3rlpy.algos.CQLConfig(compile_graph=True).create(device='cuda:0')

# train
cql.fit(
    dataset,
    n_steps=100000,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
)
