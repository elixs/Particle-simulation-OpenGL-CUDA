extern "C"
{
	void SphereWave(float3 *LocationNormal, int Segments, int Rings, float Radius, float Time);

	void AddCUDAArrays(float *First, float* Second, unsigned int Number);

	void UpdateLocation(float* Location, float* Velocity, float* Acceleration, unsigned int Number);

	void SetCUDAArray(float3* Coordinates, float3 Vector, unsigned int Number);

	void CheckBoundaries(float3* Location, float3* Velocity, unsigned int Number, float Radius, float Limit, float Friction);

	void ApplyAttractor(float3* Location, float3* Acceleration, float3 AttractorLocation, float AttractorGravity, unsigned int Number);

	void CheckCollisions(float3* Location, float3* Velocity, float3* Acceleration, unsigned int Number, float Radius, float Spring, float Damping, float Shear, float Attraction);

	void CheckLifespan(float3* Location, float3* Velocity, float* Lifespan, float4* Color, unsigned int Number);

	void RandomCUDAArray(float* Array, float Minimum, float Maximum, unsigned int Number);
}