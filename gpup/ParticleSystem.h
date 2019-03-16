#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Shader.h"

#include "Defines.h"

class GParticleSystem
{
public:
	GParticleSystem(unsigned int InNumber);
	~GParticleSystem();

	void InitGLCUDA();
	void Display(glm::mat4 Projection, glm::mat4 View, glm::vec2 LightDirection, glm::vec3 ViewDirection, float PointScale);

	void Update(bool bUseCPU);
	float* GetFirstLocation();

	void SetRandomVelocity(glm::vec3 Minimum, glm::vec3 Maximium);
	void SetLocation(glm::vec3 InLocation);
	void UpdateBoundaries(bool InbUseBoundaries, float InBoundarySize, float InBoundaryFriction);
	void UpdateGravity(bool InbUseGravity, glm::vec3 InGravity);
	void UpdateCollisions(bool InbUseCollisions, float InCollisionsSpring, float InCollisionsDamping, float InCollisionsShear, float InCollisionsAttraction);
	void UpdateAttractor(bool InbUseAttractor, glm::vec3 InAttractorLocation, float InAttractorGravity);
	void ResetLocation();
	void Reset();

private:
	unsigned int Number;
	float* Location;
	float* Velocity;
	float* Acceleration;
	float* Color;
	//float* Lifespan;
	//float* Mass;

	bool bUseBoundaries;
	float BoundarySize = BOX_SIZE;
	float BoundaryFriction;

	bool bUseGravity;
	float3 Gravity;

	bool bUseAttractor;
	float3 AttractorLocation;
	float AttractorGravity = 0.001f;
	float3 RepellerLocation;

	bool bUseCollisions;
	float CollisionsSpring;
	float CollisionsDamping;
	float CollisionsShear;
	float CollisionsAttraction;

	GShader Shader;
	unsigned int VAO;
	unsigned int LocationVBO;
	unsigned int ColorVBO;
	cudaGraphicsResource_t CUDALocationVBO;
	cudaGraphicsResource_t CUDAColorVBO;

	GShader CubeShader;
	unsigned int CubeVAO, CubeVBO, CubeEBO;
	int CubeIndicesSize;

	GShader AttractorShader;
	unsigned int AttractorVAO;
	unsigned int AttractorVBO;

	float Radius = 0.125f * 0.5f;

	bool bResetLocation = false;

	void GParticleSystem::SetCUDAArrayCPU(float3* Coordinates, float3 Vector, unsigned int Number);
	void GParticleSystem::RandomCUDAArrayCPU(float* Array, float Minimum, float Maximum, unsigned int Number);
	void GParticleSystem::ApplyAttractorCPU(float3* Location, float3* Acceleration, float3 AttractorLocation, float AttractorGravity, unsigned int Number);
	void GParticleSystem::CheckBoundariesCPU(float3* Location, float3* Velocity, unsigned int Number, float Radius, float Limit, float Friction);
	void GParticleSystem::UpdateLocationCPU(float* Location, float* Velocity, float* Acceleration, unsigned int Number);
	void GParticleSystem::CheckCollisionsCPU(float3* Location, float3* Velocity, float3* Acceleration, unsigned int Number, float Radius, float Spring, float Damping, float Shear, float Attraction);
};