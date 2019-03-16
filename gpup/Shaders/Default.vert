#version 330 core

layout (location = 0) in vec3 VLocation;
layout (location = 1) in vec3 VNormal;

uniform mat4 UModel;
uniform mat4 UView;
uniform mat4 UProjection;

out vec3 FLocation;
out vec3 FNormal;

void main()
{
	FLocation = vec3(UModel * vec4(VLocation, 1.f));
	FNormal = mat3(transpose(inverse(UModel))) * VNormal;

	gl_Position = UProjection * UView * UModel * vec4(VLocation, 1.f);
}