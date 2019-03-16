#version 330 core

layout (location = 0) in vec3 VLocation;
layout (location = 1) in vec4 VColor;

uniform mat4 UModel;
uniform mat4 UView;
uniform mat4 UProjection;

uniform float UPointRadius;
uniform float UPointScale;

out vec4 FColor;

void main()
{	
	FColor = VColor;
	gl_Position = UProjection * UView * UModel * vec4(VLocation, 1.f);
	gl_PointSize = UPointRadius * (UPointScale / length(gl_Position));
}