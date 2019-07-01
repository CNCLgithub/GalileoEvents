Documentation for galileo_ramp
******************************

.. automodule:: galileo_ramp


world
=====


.. automodule:: galileo_ramp.world
   :members:


world.scene
-----------

.. automodule:: galileo_ramp.world.scene
   :members:

world.render
------------

Rendering is primarily done by :py:mod:`galileo_ramp.world.render.render`
which cannot be called directly as it must be instatiated from within
Blender's python context.

Thus the main interface is :mod:`render.interface` which takes serialized
`dict` structures describing the world's contents (objects etc..).

There are several "data" files including textures (located at `.Textures`)
and blend files used as default parameters for lighting and resolution.

.. automodule:: galileo_ramp.world.render
   :members:

world.render.render
+++++++++++++++++++
.. automodule:: galileo_ramp.world.render.render
   :members:

world.render.interface
++++++++++++++++++++++
.. automodule:: galileo_ramp.world.render.interface
   :members:

