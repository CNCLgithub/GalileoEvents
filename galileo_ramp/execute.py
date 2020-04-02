import os


path = os.path.abspath(__file__)
mod_path = os.path.dirname(path)
img_path = os.path.join(mod_path, "sys.so")

def initialize():
    """ Initializes the Gen inference module
    :param module_path: path to inference module
    :return a function that runs inference
    """
    from julia import Julia
    jl = Julia(sysimage=img_path)

    # from julia.api import LibJulia
    # api = LibJulia.load()
    # api.sysimage = img_path
    # api.init_julia()

    from julia import GalileoRamp

    # Import the julia interface
    # from julia.api import Julia
    # jl = Julia(sysimage="sys.so", compiled_modules=False)
    # jl.eval("@eval Main import Base.MainInclude: include")
    # from julia import Main

    # Main.include("GalileoRamp")
    return GalileoRamp
