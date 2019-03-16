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

struct FDirectionalLight {
    vec3 Direction;

    FLight Light;
};
uniform FDirectionalLight UDirectionalLight;

struct FPointLight {    
    vec3 Location;

	float Constant;
	float Linear;
	float Quadratic;
  
    FLight Light;
};  
#define POINT_LIGHTS 1  
uniform FPointLight UPointLights[POINT_LIGHTS];

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

vec3 CalculateDirectonalLight(FDirectionalLight DirectionalLight, vec3 Normal, vec3 ViewDirection);
vec3 CalculatePointLight(FPointLight PointLight, vec3 Normal, vec3 FLocation, vec3 ViewDirection);
vec3 CalculateSpotLight(FSpotLight Light, vec3 Normal, vec3 FLocation, vec3 ViewDirection);
void CalculateLight(FLight SpotLight, vec3 Normal, vec3 LightDirection, vec3 ViewDirection, out vec3 Ambient, out vec3 Diffuse, out vec3 Specular);

void main()
{
	vec3 Normal = normalize(FNormal);
	vec3 ViewDirection = normalize(UViewLocation - FLocation);

	vec3 Result = CalculateDirectonalLight(UDirectionalLight, Normal, ViewDirection);

	for(int i = 0; i < POINT_LIGHTS; ++i)
	{
		Result += CalculatePointLight(UPointLights[i], Normal, FLocation, ViewDirection);
	}

	Result += CalculateSpotLight(USpotLight, Normal, FLocation, ViewDirection);

	OFragColor = vec4(Result, 1.f);
}

vec3 CalculateDirectonalLight(FDirectionalLight DirectionalLight, vec3 Normal, vec3 ViewDirection)
{
    vec3 LightDirection = normalize(-DirectionalLight.Direction);

    vec3 Ambient, Diffuse, Specular;
	CalculateLight(DirectionalLight.Light, Normal, LightDirection, ViewDirection, Ambient, Diffuse, Specular);

    return  Ambient + Diffuse + Specular;
}

vec3 CalculatePointLight(FPointLight PointLight, vec3 Normal, vec3 FLocation, vec3 ViewDirection)
{
	vec3 LightDirection = normalize(PointLight.Location - FLocation);

    vec3 Ambient, Diffuse, Specular;
    CalculateLight(PointLight.Light, Normal, LightDirection, ViewDirection, Ambient, Diffuse, Specular);

	float Distance = length(PointLight.Location - FLocation);
	float Attenuation = 1.0 / (PointLight.Constant + PointLight.Linear * Distance + PointLight.Quadratic * (Distance * Distance));  

	Ambient *= Attenuation;
	Diffuse *= Attenuation;
	Specular *= Attenuation;

	return Ambient + Diffuse + Specular;
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

