run_exp1_trial("/databases/exp1.hdf5", 1, 1, nothing);
println("finished first run");
# @time run_exp1_trial("/databases/exp1.hdf5", 1, 1, nothing);
# @time run_exp1_trial("/databases/exp1.hdf5", 1, 10, nothing);
# @time run_exp1_trial("/databases/exp1.hdf5", 1, 100, nothing);
@time run_exp1_trial("/databases/exp1.hdf5", 1, 10, "test.jld2");
# @time run_exp1_trial("/databases/exp1.hdf5", 1, 100, "test.jld2");
# @time run_exp1_trial("/databases/exp1.hdf5", 1, 1000, "test.jld2");
