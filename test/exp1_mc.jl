# Inference Core - (src/queries/exp1.jl)
# compile first pass
seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = nothing);
# @time seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = "test.jld2");
# # @time seq_inference("/databases/exp1.hdf5", 1, 10, 0.1; out = "test.jld2");

# # Inference api for BO - (src/queries/exp1.jl)
# @time seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; bo = true);
# results = @time seq_inference("/databases/exp1.hdf5", 1, 2, 0.1; bo = true);

# # Analysis Core - (src/analysis.jl)
# extracted = @time extract_chain(results);
# df = @time to_frame(extracted["log_scores"], extracted["unweighted"]);
# df = @time digest_pf_trial(results, [60, 70, 80, 90])
a = @time evaluation(0.1, 1, "/databases/exp1.hdf5", 0);
b = @time evaluation(0.1, 1, "/databases/exp1.hdf5", 120);
rmse = merge_evaluation([a,b], "/databases/exp1_avg_human_responses.csv")
