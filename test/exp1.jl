seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = nothing);
println("finished first run");
@time seq_inference("/databases/exp1.hdf5", 1, 1, 0.1; out = "test.jld2");
@time seq_inference("/databases/exp1.hdf5", 1, 10, 0.1; out = "test.jld2");
