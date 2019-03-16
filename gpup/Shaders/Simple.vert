#version 330 core

layout (location = 0) in vec3 VLocation;

void main()
{
	gl_Position = vec4(VLocation, 1.f);
}