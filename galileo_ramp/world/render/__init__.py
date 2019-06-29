""" Describes aspects of the Generative model related to rendering.

Overview:

Rendering is primarily done by :mod: `render.render`
which cannot be called directly as it must be instatiated from within
Blender's python context.

Thus the main interface is :mod: `render.interface` which takes serialized
`dict` structures describing the world's contents (objects etc..).

There are several "data" files including textures (located at `.Textures`)
and blend files used as default parameters for lighting and resolution.

"""
