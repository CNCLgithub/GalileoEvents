# Inference Core - (src/queries/exp1.jl)
# compile first pass
# seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = nothing);
# @time seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = nothing);
# @time seq_inference("/databases/exp1.hdf5", 1, 10, 0.1; out = nothing);
# @time seq_inference("/databases/exp1.hdf5", 1, 100, 0.1; out = nothing);

# # Inference api for BO - (src/queries/exp1.jl)
# results = @time seq_inference("/databases/exp1.hdf5", 1, 10, 0.1; bo = true);

# # Analysis Core - (src/analysis.jl)
# extracted = @time extract_chain(results);
# df = @time to_frame(extracted["log_scores"], extracted["unweighted"]);
# df = @time digest_pf_trial(results, [60, 70, 80, 90])
# println(df)
a = @time evaluation(0.1, 10, "/databases/exp1.hdf5", 0, reps = 2);
b = @time evaluation(0.1, 10, "/databases/exp1.hdf5", 1, reps = 2);
rmse = merge_evaluation([a,b], "/databases/exp1_avg_human_responses.csv")
