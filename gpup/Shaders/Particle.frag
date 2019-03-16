#version 330 core

uniform vec3 ULightDirection;
uniform vec3 UViewDirection;

in vec4 FColor;

out vec4 OFragColor;

void main()
{
	vec2 CircleCoord = gl_PointCoord * vec2(2.f,-2.f) + vec2(-1.f, 1.f);

	float Distance = dot(CircleCoord, CircleCoord); 
	float Alpha = 1.f - smoothstep(0.97f, 1.f, Distance);
	if(Alpha < 0.5f)
	{
		discard;
	}

	// Find rotation matrix from Start to UViewDirection
	vec3 Start = vec3(0.f, 0.f, -1.f);
	vec3 Cross = cross(UViewDirection, Start);
	float Sine = length(Cross);
	float Cosine = dot(Start, UViewDirection);
	mat3 CrossMatrix = mat3(0.f, Cross[2], -Cross[1], -Cross[2], 0.f, Cross[0], Cross[1], -Cross[0], 0.f);
	mat3 RotationMatrix = mat3(1.f) + CrossMatrix + ((1 - Cosine)/(Sine * Sine))*(CrossMatrix * CrossMatrix);

	vec3 Shadow = vec3(CircleCoord, sqrt(1 - Distance));
	float ShadowRatio = max(0.f, dot((RotationMatrix * -ULightDirection), Shadow));

	OFragColor = vec4(FColor.rgb * ShadowRatio, FColor.a * Alpha);
}