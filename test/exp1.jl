run_exp1_trial("/databases/exp1.hdf5", 1, 1, 0.1, nothing);
println("finished first run");
@time run_exp1_trial("/databases/exp1.hdf5", 1, 1, 0.1, "test.jld2");
@time run_exp1_trial("/databases/exp1.hdf5", 1, 10, 0.1, "test.jld2");
