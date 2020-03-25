"""Extracts inference traces from chains for analysis"""

using GalileoRamp

density_map = Dict(
    "density" => cm -> cm[:object_physics => 1 => :density]
)

function parse_chain(chain_path)
    chain = ...
    e = extract_chain(trace, density_map)

end

function main():
    dataset_path = "/databases/exp1.hdf5"
    dataset = ...
    traces = "/traces/exp1"
    for i = 1:length(dataset)
        scene,_,tps = get(dataset, i)
        trace_path = "$traces/$i.jld2"
    end
end


main();
