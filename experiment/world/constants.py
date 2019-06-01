# x,y,z
dimensions = np.array([3.0, 3.0, 1.5])

materials = ["Iron", "Brick", "Wood"]

'''
 DENSITY ratio according to engineering toolbox
    iron = 7800 kg/m^3    | 11.14
    brick = 2000 kg/m^3    | 2.857
    wood = 700 kg/m^3    | 1.000
'''
dens = np.array([8.0, 2.0, 1.0]) # / np.prod(dimensions)
densities = dict(zip(materials, dens))
density_bounds = [0.01, 2000.0]


'''

 FRICTION ratio according to engineering toolbox
    brick -> wood = 0.6
    wood -> iron = 0.4
    __

     brick = 0.6 / wood
    iron = (2/3) * Brick*

taken from http://www.engineeringtoolbox.com/friction-coefficients-d_778.html
on 10/10/17
'''
frics = np.array([0.215, 0.323, 0.263])
frictions = dict(zip(materials, frics))
friction_bounds = [0.001, 1.0]

shapes = ["Block", "Puck"]