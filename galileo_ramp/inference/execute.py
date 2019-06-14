import os

def initialize(module_path):
    """ Initializes the Gen inference module
    :param module_path: path to inference module
    :return a function that runs inference
    """
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            "Module at {0!s} not found!".format(module_path)
            )
    # Import the julia interface
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    jl.eval("@eval Main import Base.MainInclude: include")
    from julia import Main

    Main.include(module_path)
    return Main.run_inference
