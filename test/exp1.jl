seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = nothing);
println("finished first run");
@time seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = "test.jld2");
# @time seq_inference("/databases/exp1.hdf5", 1, 10, 0.1; out = "test.jld2");
@time seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; bo = true);
results = @time seq_inference("/databases/exp1.hdf5", 1, 10, 0.1; bo = true);
extracted = @time extract_chain(results);
df = @time to_frame(extracted["log_scores"], extracted["unweighted"]);
df = @time digest_pf_trial(results, [60, 70, 80, 90])
