#version 330 core

uniform vec3 ULocation;

uniform mat4 UModel;
uniform mat4 UView;
uniform mat4 UProjection;

uniform float UPointSize;
uniform float UPointScale;

out vec4 FColor;

void main()
{	
	gl_Position = UProjection * UView * UModel * vec4(ULocation, 1.f);
	if(UPointSize < 0)
	{
		FColor = vec4(1.f);
		gl_PointSize = 100.f * -UPointSize * (UPointScale / length(gl_Position));
	}
	else
	{
		FColor = vec4(vec3(0.f), 1.f);
		gl_PointSize = 100.f * UPointSize * (UPointScale / length(gl_Position));
	}
}