from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed
# example objective taken from skopt
from skopt.benchmarks import branin

optimizer = Optimizer(
    dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
    random_state=1,
    base_estimator='gp'
)

for i in range(5):
    x = optimizer.ask(n_points=5)
    print(x)  # x is a list of n_points points
    y = Parallel(n_jobs=15)(delayed(branin)(v) for v in x)  # evaluate points in parallel
    optimizer.tell(x, y)

# takes ~ 20 sec to get here
print(min(optimizer.yi))  # print the best objective found
a = optimizer.yi.index(min(optimizer.yi))
print(optimizer.yi.index(min(optimizer.yi)))  # print the best objective found
print(optimizer.Xi[a])  # print the best objective founda