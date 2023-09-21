export ramp

# TODO: support the interface defined in the docstring below
"""
    ramp(mass_ratio::Float64, obj_frictions::NTuple{2, Float64},
         obj_positions::NTuple{2}; slope, ramp_intersection)
"""
function ramp(
    slope::Float64=2/3,
    tableRampIntersection::Float64=0.,
    )
    # for debugging
    #client = @pycall pb.connect(pb.GUI)::Int64
    #pb.resetDebugVisualizerCamera(4.5, 0, -40, [0.0, 0.0, 0.0]; physicsClientId=client)
    client = @pycall pb.connect(pb.DIRECT)::Int64
    pb.setGravity(0,0,-10; physicsClientId = client)


    # add a table base
    grey = [0.5, 0.5, 0.5, 1]
    base_dims = [5, 1, 0.75] # in meters
    table_dims = [base_dims[1] + 0.2, base_dims[2] + 0.2, 0.1]  # Width, depth, height
    table_base_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents = base_dims / 2, physicsClientId = client)
    table_base_obj_id = pb.createMultiBody(baseCollisionShapeIndex = table_base_col_id, basePosition = [0,0,-(base_dims[3]+table_dims[3])/2], physicsClientId = client)
    pb.changeDynamics(table_base_obj_id, -1; mass = 0., restitution = 0.9, physicsClientId=client)
    pb.changeVisualShape(table_base_obj_id, -1, rgbaColor=grey, physicsClientId=client)

    # Create the tabletop (a flat box)
    table_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=table_dims/2)
    table_body_id = pb.createMultiBody(baseCollisionShapeIndex=table_col_id, basePosition=[0, 0, -table_dims[3]/2])
    pb.changeDynamics(table_body_id, -1; mass = 0., restitution = 0.9, physicsClientId=client)
    pb.changeVisualShape(table_body_id, -1, rgbaColor=grey.+0.2, physicsClientId=client)

    # Create the four frame-like boxes around the tabletop
    frame_height = 0.25
    frame_thickness = 0.05

    frame_dims = [
        [table_dims[1] + 2 * frame_thickness, frame_thickness, frame_height],  # Longer sides
        [table_dims[1] + 2 * frame_thickness, frame_thickness, frame_height],  # Longer sides
        [frame_thickness, table_dims[2], frame_height],  # Shorter sides
        [frame_thickness, table_dims[2], frame_height]  # Shorter sides
    ]

    frame_positions = [
        [0, table_dims[2] / 2 + frame_thickness / 2, 0],  # Top side
        [0, -table_dims[2] / 2 - frame_thickness / 2, 0],  # Bottom side
        [table_dims[1] / 2 + frame_thickness / 2, 0, 0],  # Right side
        [-table_dims[1] / 2 - frame_thickness / 2, 0, 0]  # Left side
    ]

    for (dims, pos) in zip(frame_dims, frame_positions)
        frame_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=dims/2)::Int64
        frame_obj_id = pb.createMultiBody(baseCollisionShapeIndex=frame_col_id, basePosition=pos)::Int64
        pb.changeVisualShape(frame_obj_id, -1, rgbaColor=grey, physicsClientId=client)
    end

    # add a ramp
    ramp_col_id = pb.createCollisionShape(pb.GEOM_MESH, fileName="examples/ramp/ramp.obj", physicsClientId=client, meshScale=[2, base_dims[2], slope*2])
    ramp_position = [-2+tableRampIntersection, -base_dims[2]/2, 0]
    ramp_obj_id = pb.createMultiBody(baseCollisionShapeIndex=ramp_col_id, basePosition=ramp_position, physicsClientId=client)
    pb.changeDynamics(ramp_obj_id, -1; mass=0.0, restitution=0.9, physicsClientId=client)
    pb.changeVisualShape(ramp_obj_id, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=client)

    # add a floor
    floor_col_id = pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=client)
    floor_obj_id = pb.createMultiBody(baseCollisionShapeIndex=floor_col_id, basePosition=[0,0,-base_dims[3]], physicsClientId=client)
    pb.changeDynamics(floor_obj_id, -1; mass=0.0, restitution=0.9, physicsClientId=client)


    #  add walls
    wall_dims = [[0.1, 8.0, 5.0], [0.1, 8.0, 5.0], [8.0, 0.1, 5.0]] # Width, length, height
    wall_positions = [
        [4.0, 0.0, 1.0],  # Right Wall
        [-4.0, 0.0, 1.0],  # Left Wall
        [0, 4, wall_dims[3][3]/2-base_dims[3]] # Back Wall
    ]
    for (dims, pos) in zip(wall_dims, wall_positions)
        wall_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=dims./2, physicsClientId=client)
        wall_obj_id = pb.createMultiBody(baseCollisionShapeIndex=wall_col_id, basePosition=pos, physicsClientId=client)
        pb.changeDynamics(wall_obj_id, -1; mass=0.0, restitution=0.9, physicsClientId=client)
        pb.changeVisualShape(wall_obj_id, -1, rgbaColor=grey+[0.2, 0.2, 0.2, 0], physicsClientId=client)
    end

    # add an object on the ramp
    obj_ramp_dims = [0.15, 0.3, 0.075]
    theta_radians = -atan(slope)
    orientation = [cos(theta_radians / 2), 0, sin(theta_radians / 2), 0]

    obj_on_ramp_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj_ramp_dims/2, physicsClientId=client)
    lift = obj_ramp_dims[3]/2
    position = [
        -1+tableRampIntersection+lift*cos(theta_radians),
        0,
        1*slope-lift*sin(theta_radians)
    ]
    obj_on_ramp_obj_id = pb.createMultiBody(baseCollisionShapeIndex=obj_on_ramp_col_id, basePosition=position, baseOrientation=orientation, physicsClientId=client)
    pb.changeDynamics(obj_on_ramp_obj_id, -1; mass=1.0, restitution=0.9, physicsClientId=client)

    # add an object on the table that will collide with the object on the ramp as that one slides down
    obj_on_table_dims = [0.2, 0.2, 0.1]
    obj_on_table_col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj_on_table_dims/2, physicsClientId=client)
    obj_on_table_obj_id = pb.createMultiBody(baseCollisionShapeIndex=obj_on_table_col_id, basePosition=[1, 0, obj_on_table_dims[3]/2], physicsClientId=client)
    pb.changeDynamics(obj_on_table_obj_id, -1; mass=1.0, restitution=0.9, physicsClientId=client)

    (client, obj_on_ramp_obj_id, obj_on_table_obj_id)
end

