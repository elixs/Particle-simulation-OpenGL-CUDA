#version 330 core

struct FMaterial {
    vec3 Ambient;
	vec3 Diffuse;
	vec3 Specular;
    float Shininess;
}; 
uniform FMaterial UMaterial;

struct FLight {

    vec3 Ambient;
    vec3 Diffuse;
    vec3 Specular;
};

struct FSpotLight {
    vec3 Location;
    vec3 Direction;

	float Constant;
	float Linear;
	float Quadratic;

	float CutOff;
	float OuterCutOff;

    FLight Light;
};
uniform FSpotLight USpotLight;

uniform vec3 UViewLocation;

in vec3 FLocation;
in vec3 FNormal;

out vec4 OFragColor;

vec3 CalculateSpotLight(FSpotLight Light, vec3 Normal, vec3 FLocation, vec3 ViewDirection);
void CalculateLight(FLight SpotLight, vec3 Normal, vec3 LightDirection, vec3 ViewDirection, out vec3 Ambient, out vec3 Diffuse, out vec3 Specular);

void main()
{
	vec3 Normal = normalize(FNormal);
	vec3 ViewDirection = normalize(UViewLocation - FLocation);

	vec3 Result = CalculateSpotLight(USpotLight, Normal, FLocation, ViewDirection);

	OFragColor = vec4(5.f * Result, 1.f);
}

vec3 CalculateSpotLight(FSpotLight SpotLight, vec3 Normal, vec3 FLocation, vec3 ViewDirection)
{
	vec3 LightDirection = normalize(SpotLight.Location - FLocation);

    vec3 Ambient, Diffuse, Specular;
    CalculateLight(SpotLight.Light, Normal, LightDirection, ViewDirection, Ambient, Diffuse, Specular);

	float Distance = length(SpotLight.Location - FLocation);
	float Attenuation = 1.0 / (SpotLight.Constant + SpotLight.Linear * Distance + SpotLight.Quadratic * (Distance * Distance));

	float Theta = dot(LightDirection, normalize(-SpotLight.Direction));
	float Epsilon   = SpotLight.CutOff - SpotLight.OuterCutOff;
	float Intensity = clamp((Theta - SpotLight.OuterCutOff) / Epsilon, 0.0, 1.0);

	Ambient *= Attenuation * Intensity;
	Diffuse *= Attenuation * Intensity;
	Specular *= Attenuation * Intensity;

	return Ambient + Diffuse + Specular;
}

void CalculateLight(FLight Light, vec3 Normal, vec3 LightDirection, vec3 ViewDirection, out vec3 Ambient, out vec3 Diffuse, out vec3 Specular)
{
	float DiffuseRatio = max(dot(Normal, LightDirection), 0.f);
    vec3 ReflectionDirection = reflect(-LightDirection, Normal);
    float SpecularRatio = pow(max(dot(ViewDirection, ReflectionDirection), 0.f), UMaterial.Shininess);

    Ambient  = Light.Ambient  * UMaterial.Ambient;
    Diffuse  = Light.Diffuse  * DiffuseRatio * UMaterial.Diffuse;
    Specular = Light.Specular * SpecularRatio * UMaterial.Specular;
}