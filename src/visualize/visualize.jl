export visualize

using Luxor

### DEFINE CONSTANTS ###

# Define color map for different materials
color_dict = Dict{String,String}("Iron" => "silver",
                                 "Wood" => "burlywood",
                                 "Brick" => "firebrick",
                                 "ramp" => "black",
                                 "table" => "black")

# global scaling
scale_fac = -100

########################


function draw_box(obj_data, pos, orientation)
    pos = pos*scale_fac
    dim = obj_data["dims"]*scale_fac
    sethue(color_dict[obj_data["appearance"]])
    gsave()
    translate(pos)
    rotate(-orientation)
    box(O, dim[1], dim[3], :stroke)
    grestore()
end

function draw_groundtruth(scene_data, obj_positions)
    data = deepcopy(scene_data)
    # add ramp and table
    obj = data["table"]
    pos = Point(obj["position"][1], obj["position"][3])
    orientation = obj["orientation"][2]
    draw_box(obj, pos, orientation)
    obj = data["ramp"]
    pos = Point(obj["position"][1], obj["position"][3])
    orientation = obj["orientation"][2]
    draw_box(obj, pos, orientation)
    # loop through objects
    objects = data["objects"]
    for (idx, item) in enumerate(sort(collect(keys(objects))))
        obj = objects[item]
        pos = Point(obj_positions[idx, 1], obj_positions[idx, 3])
        draw_box(obj, pos, 0)
    end
end

function draw_predictions(predictions, gt)
    color = gt ? "green" : "blue"
    scale = gt ? 3.5 : 2.5
    num_sids = first(size(predictions))
    for sid = 1:num_sids
        sethue(color)
        Luxor.circle(predictions[sid, 1, 1]*scale_fac,
                     predictions[sid, 1, 3]*scale_fac,
                     scale, :stroke)
        Luxor.circle(predictions[sid, 2, 1]*scale_fac,
                     predictions[sid, 2, 3]*scale_fac,
                     scale, :stroke)
    end
end


function visualize(gt,
                   observations::T,
                   predictions::Array{Float64, 4},
                   gt_sim::Union{T,Nothing},
                   path::String) where {T<:Array{Float64, 3}}
    scene_length = first(size(observations))
    mov = Movie(750, 300, "visualization", 1:scene_length)
    backdrop(scene, framenumber) = background("white")
    function frame(scene, framenumber)
        Luxor.origin()
        translate(Point(0.,100.))
        draw_groundtruth(gt, observations[framenumber, :, :])
        draw_predictions(predictions[framenumber, :, :, :], false)
        isnothing(gt_sim) || draw_predictions(gt_sim[:, framenumber, :, :], true)
    end
    animate(mov,
            [Luxor.Scene(mov, backdrop, 1:scene_length),
             Luxor.Scene(mov, frame, 1:scene_length)],
            creategif=true,
            usenewffmpeg=false,
            pathname=path)
end




