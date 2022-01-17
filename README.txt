
Neuron Feature Comparison Tools

The set of tools should support through a process pipeline the extraction of neuron features and using 
dimension reduction, heuristics, and classification evaluate correctness of neuron reconstructions. 
Features include synapses, volume, area, among others. Using these strategies, screen neuronal 
reconstructions for candidates that need further proofreading.

Since there are different incremental tasks involved in the feature extraction pipeline, the code is 
roughly broken down in those individual tasks:

- Comptoolconst: includes all names and constants.
- create_feature_skeleton: Sets up the feature dataframe and fills it with the nucleus data and their partitions.
- comparison_tool: performs the actual data queries for each root id and populates the feature columns.
- extract_proofed_data: simply extracts and stores the current proof data. 
- setup_ground_truth: merges proof date with feature information to create the consolidated dataset.
- correlation_visuals: performs different dimension reduction and visualization tasks for data analysis. The visualization will be placed in 
  a separate library.



