//#pragma once

#include <curand_kernel.h>
#define PI 3.14159265358979323846

inline __host__ __device__ float3 operator+(const float3 &A, const float3 &B)
{
	return make_float3(A.x + B.x, A.y + B.y, A.z + B.z);
}

inline __host__ __device__ float3 operator-(const float3 &A, const float3 &B)
{
	return make_float3(A.x - B.x, A.y - B.y, A.z - B.z);
}

inline __host__ __device__ float3 operator-(const float3 &A)
{
	return make_float3(-A.x, -A.y, -A.z);
}

inline __host__ __device__ float3 operator+(float S, float3 A)
{
	return make_float3(S + A.x, S + A.y, S + A.z);
}

inline __host__ __device__ float3 operator-(float S, float3 A)
{
	return make_float3(S - A.x, S - A.y, S - A.z);
}

inline __host__ __device__ void operator+=(float3 &A, float3 B)
{
	A.x += B.x; A.y += B.y; A.z += B.z;
}

inline __host__ __device__ void operator-=(float3 &A, float3 B)
{
	A.x -= B.x; A.y -= B.y; A.z -= B.z;
}

inline __host__ __device__ float Dot(float3 A, float3 B)
{
	return A.x * B.x + A.y * B.y + A.z * B.z;
}

inline __host__ __device__ float3 operator*(float3 A, float3 B)
{
	return make_float3(A.x * B.x, A.y * B.y, A.z * B.z);
}

inline __host__ __device__ float3 operator*(float3 A, float S)
{
	return make_float3(A.x * S, A.y * S, A.z * S);
}

inline __host__ __device__ float3 operator*(float S, float3 A)
{
	return make_float3(A.x * S, A.y * S, A.z * S);
}

inline __host__ __device__ void operator*=(float3 &A, float S)
{
	A.x *= S; A.y *= S; A.z *= S;
}

inline __host__ __device__ float3 operator/(float3 A, float3 B)
{
	return make_float3(A.x / B.x, A.y / B.y, A.z / B.z);
}

inline __host__ __device__ float3 operator/(float3 A, float S)
{
	return make_float3(A.x / S, A.y / S, A.z / S);
}

inline __host__ __device__ float Length(float3 V)
{
	return sqrtf(Dot(V, V));
}

inline __host__ __device__ float3 Normalize(float3 V)
{
	return rsqrtf(Dot(V, V))*V;
}

inline __host__ __device__ float Clamp(float X, float A, float B)
{
	return max(A, min(B, X));
}

__global__ void SphereKernel(float3 *LocationNormal, int Segments, int Rings, float Radius, float Time)
{
	unsigned int X = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int Y = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned int Id = (Y * Segments + X + 2) * 2;

	if (Id < 2 * (Segments * (Rings - 1) + 2))
	{
		// calculate uv coordinates
		float U = X / (float)Segments;
		float V = Y / (float)Rings;
		U = U * 2.0f - 1.0f;
		V = V * 2.0f - 1.0f;

		// calculate simple sine wave pattern
		float Frequency = PI;
		float W = sinf(U * Frequency + Time) * cosf(V * Frequency + Time);

		// write output vertex
		LocationNormal[Id] = W + Radius * LocationNormal[Id + 1];
	}
}

__global__ void AddKernel(float *First, float *Second, unsigned int Number)
{
	unsigned int Id = blockIdx.x*blockDim.x + threadIdx.x;

	if (Id < Number)
	{
		Second[Id] += First[Id];
	}
}

__global__ void UpdateKernel(float* Location, float* Velocity, float* Acceleration, unsigned int Number)
{
	unsigned int Id = blockIdx.x*blockDim.x + threadIdx.x;

	if (Id < Number)
	{
		Velocity[Id] += Acceleration[Id];
		Location[Id] += Velocity[Id];
	}
}

__global__ void SetVectorKernel(float3* Coordinates, float3 Vector, unsigned int Number)
{
	unsigned int Id = blockIdx.x*blockDim.x + threadIdx.x;

	if (Id < Number)
	{
		Coordinates[Id] = Vector;
	}
}

__global__ void BoundaryKernel(float3* Location, float3* Velocity, unsigned int Number, float Radius, float Limit, float Friction)
{
	unsigned int Id = blockIdx.x*blockDim.x + threadIdx.x;

	if (Id < Number)
	{
		if (Location[Id].x < -Limit + Radius)
		{
			Location[Id].x = -Limit + Radius;
			Velocity[Id].x *= -(1.f - Friction);
			Velocity[Id].y *= (1.f - Friction);
			Velocity[Id].z *= (1.f - Friction);
		}
		if (Location[Id].x > Limit - Radius)
		{
			Location[Id].x = Limit - Radius;
			Velocity[Id].x *= -(1.f - Friction);
			Velocity[Id].y *= (1.f - Friction);
			Velocity[Id].z *= (1.f - Friction);
		}
		if (Location[Id].y < -Limit + Radius)
		{
			Location[Id].y = -Limit + Radius;
			Velocity[Id].x *= (1.f - Friction);
			Velocity[Id].y *= -(1.f - Friction);
			Velocity[Id].z *= (1.f - Friction);
		}
		if (Location[Id].y > Limit - Radius)
		{
			Location[Id].y = Limit - Radius;
			Velocity[Id].x *= (1.f - Friction);
			Velocity[Id].y *= -(1.f - Friction);
			Velocity[Id].z *= (1.f - Friction);
		}
		if (Location[Id].z < -Limit + Radius)
		{
			Location[Id].z = -Limit + Radius;
			Velocity[Id].x *= (1.f - Friction);
			Velocity[Id].y *= (1.f - Friction);
			Velocity[Id].z *= -(1.f - Friction);
		}
		if (Location[Id].z > Limit - Radius)
		{
			Location[Id].z = Limit - Radius;
			Velocity[Id].x *= (1.f - Friction);
			Velocity[Id].y *= (1.f - Friction);
			Velocity[Id].z *= -(1.f - Friction);
		}
	}
}

__device__ float CUDARandFloat(float Seed, float Minimum, float Maximum)
{
	curandState_t State;
	curand_init(clock64(), Seed, 0, &State);
	return Minimum + curand_uniform(&State) * (Maximum - Minimum);
}

__global__ void AttractorKernel(float3* Location, float3* Acceleration, float3 AttractorLocation, float AttractorGravity, unsigned int Number)
{
	unsigned int Id = blockIdx.x*blockDim.x + threadIdx.x;

	if (Id < Number)
	{
		float3 Force = AttractorLocation - Location[Id];
		float Distance = Clamp(Length(Force), 1.f, 5.f);

		Force = Normalize(Force);
		float Strength = AttractorGravity / (Distance * Distance);
		Force = Force * Strength;
		Acceleration[Id] += Force;
	}
}

__global__ void CollisionsKernel(float3* Location, float3* Velocity, float3* Acceleration, unsigned int Number, float Radius, float Spring, float Damping, float Shear, float Attraction)
{
	unsigned int X = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int Y = blockIdx.y*blockDim.y + threadIdx.y;

	if (X != Y && X < Number && Y < Number)
	{
		float3 RelativeLocation = Location[Y] - Location[X];
		float Distance = Length(RelativeLocation);
		if (Distance < 2 * Radius)
		{
			float3 Direction = RelativeLocation / Distance;
			float3 RelativeVelocity = Velocity[Y] - Velocity[X];
			float3 TangencialVelocity = RelativeVelocity - Dot(RelativeVelocity, Direction) * Direction;


			Acceleration[X] -= Spring * (2 * Radius - Distance) * Direction;
			Acceleration[X] += Damping * RelativeVelocity;
			Acceleration[X] += Shear * TangencialVelocity;
			Acceleration[X] += Attraction * RelativeLocation;
		}
	}
}

__global__ void LifespanKernel(float3* Location, float3* Velocity, float* Lifespan, float4* Color, unsigned int Number)
{
	unsigned int Id = blockIdx.x*blockDim.x + threadIdx.x;

	if (Id < Number)
	{
		Lifespan[Id] -= 0.01f;

		if (Lifespan[Id] < 0.f)
		{
			//Lifespan[Id] = 1.f;
			Lifespan[Id] = CUDARandFloat(Id, 0.5f, 1.f);
			Location[Id] = make_float3(0.f, 1.f, 0.f);
			Velocity[Id] = make_float3(CUDARandFloat(3 * Id, -0.03f, 0.03f), CUDARandFloat(3 * Id + 1, 0.f, 0.06f), CUDARandFloat(3 * Id + 2, -0.03f, 0.03f));
		}

		Color[Id].w = Lifespan[Id];
	}
}

__global__ void RandomKernel(float* Array, float Minimum, float Maximum, unsigned int Number)
{
	unsigned int Id = blockIdx.x*blockDim.x + threadIdx.x;

	if (Id < Number)
	{
		Array[Id] = CUDARandFloat(Id, Minimum, Maximum);
	}
}