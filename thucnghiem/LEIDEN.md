# Leiden backend

The proposed algorithm and all GA variants are pure Julia. For the Leiden baseline, the project intentionally uses an optional PythonCall bridge because a stable, current Julia-native Leiden API is not part of the core Graphs.jl API.

Install in Julia:

```julia
using Pkg
Pkg.add("PythonCall")
```

Then make sure the Python environment visible to PythonCall contains `python-igraph` and `leidenalg`. If Leiden is not installed, run the main experiment with `--skip-leiden`; do not silently substitute Louvain and label it Leiden.
