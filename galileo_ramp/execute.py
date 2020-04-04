import os


path = os.path.abspath(__file__)
mod_path = os.path.dirname(path)
img_path = os.path.join(mod_path, "sys.so")

# def initialize():
#     # Import the julia interface
#     from julia.api import Julia
#     jl = Julia(compiled_modules=False)
#     jl.eval("@eval Main import GalileoRamp")
#     return jl


def initialize(init_julia = True):
    """ Initializes the Gen inference module
    :param module_path: path to inference module
    :return a function that runs inference
    """

    # Import the julia interface
    from julia.api import Julia
    jl = Julia(compiled_modules=False, init_julia = init_julia)
    jl.eval("@eval Main import GalileoRamp")
    from julia import GalileoRamp
    return GalileoRamp
    # from julia import Main

    # gr = Main.import("GalileoRamp")
    # return gr

    # from julia import Julia
    # # jl = Julia(sysimage=img_path)
    # jl = Julia(compiled_modules=False)

    # # from julia.api import LibJulia
    # # api = LibJulia.load()
    # # api.sysimage = img_path
    # # api.init_julia()

    # from julia import GalileoRamp
