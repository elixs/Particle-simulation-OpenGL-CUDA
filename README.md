Please follow the tutorial of https://github.com/elixs/Terrain-rendering in order to install de dependencies.

There is a .exe and Settings.txt to test the program without installing dependencies. Settings.txt contains a single number which indicates the number of particles to use.

The particles of this simulation are rendered using only a single dot with the help of glsl shaders in order to achieve an acceptable simulation of a directional light without the overload of extra geometry.

Application controls:

WASDQE    Movement
Mouse     Camera movement
Scroll    Zoom
Shift     Hold to move faster
Control   Hold to move slower
Space     Pause simulation
F         Toggle to fix simulated direcional light 
H         Hide info
T         OpenGL/CUDA test
R         Reset particle simulation
Esc       Access/exit interactive menu
