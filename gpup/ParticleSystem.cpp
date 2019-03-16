#include "ParticleSystem.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>

#include <glad/glad.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_gl_interop.h>
#include "CUDA_helper/helper_cuda.h"

#include "Main.cuh"
#include "Utils.h"

#include "Defines.h"

#include "resource.h"

GParticleSystem::GParticleSystem(unsigned int InNumber) : Number(InNumber)
{
	if (Number == 0)
	{
		Number = std::atoi(FileToChar("Settings.txt"));
	}
	Location = new float[3 * Number];
	Color = new float[4 * Number];
	checkCudaErrors(cudaMallocManaged(&Velocity, 3 * Number * sizeof(float)));
	checkCudaErrors(cudaMallocManaged(&Acceleration, 3 * Number * sizeof(float)));
	//checkCudaErrors(cudaMallocManaged(&Lifespan, Number * sizeof(float)));

	AttractorLocation = make_float3(0.f, 0.f, 0.f);

	for (unsigned int i = 0; i < Number; ++i)
	{
		Location[i * 3] = RandFloat(-BoundarySize / 2.f + Radius, BoundarySize / 2.f - Radius);
		Location[i * 3 + 1] = RandFloat(-BoundarySize / 2.f + Radius, BoundarySize / 2.f - Radius);
		Location[i * 3 + 2] = RandFloat(-BoundarySize / 2.f + Radius, BoundarySize / 2.f - Radius);

		Color[i * 4] = RandFloat(0.f, 1.f);
		Color[i * 4 + 1] = RandFloat(0.f, 1.f);
		Color[i * 4 + 2] = RandFloat(0.f, 1.f);
		Color[i * 4 + 3] = 1.f;

		//Lifespan[i] = RandFloat(0.1f, 1.f);
	}
#ifdef STAND_ALONE
	Shader = GShader::GShader(ParticleVert, ParticleFrag);
	CubeShader = GShader::GShader(DefaultVert, SimpleFrag);
	AttractorShader = GShader::GShader(AttractorVert, ParticleFrag);
#else
	Shader = GShader::GShader("Shaders/Particle.vert", "Shaders/Particle.frag");
	CubeShader = GShader::GShader("Shaders/Default.vert", "Shaders/Simple.frag");
	AttractorShader = GShader::GShader("Shaders/Attractor.vert", "Shaders/Particle.frag");
#endif // STAND_ALONE
}

GParticleSystem::~GParticleSystem()
{
	delete[] Location;
	cudaFree(Velocity);
	cudaFree(Acceleration);

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &LocationVBO);
	glDeleteBuffers(1, &ColorVBO);

	glDeleteVertexArrays(1, &AttractorVAO);
	glDeleteBuffers(1, &AttractorVBO);
}

void GParticleSystem::InitGLCUDA()
{
	// Particles
	glEnable(GL_PROGRAM_POINT_SIZE);

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &LocationVBO);
	glGenBuffers(1, &ColorVBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, LocationVBO);
	glBufferData(GL_ARRAY_BUFFER, 3 * Number * sizeof(float), Location, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, ColorVBO);
	glBufferData(GL_ARRAY_BUFFER, 4 * Number * sizeof(float), Color, GL_STATIC_DRAW);

	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);

	// Cube
	// in
	float* Cube;
	int CubeSize;

	// out
	int* CubeIndices;
	//int CubeIndicesSize;

	GenerateCubeWire(Cube, CubeIndices, 1.f, CubeSize, CubeIndicesSize);

	glGenVertexArrays(1, &CubeVAO);
	glGenBuffers(1, &CubeVBO);
	glGenBuffers(1, &CubeEBO);

	glBindVertexArray(CubeVAO);

	glBindBuffer(GL_ARRAY_BUFFER, CubeVBO);
	glBufferData(GL_ARRAY_BUFFER, CubeSize * sizeof(float), Cube, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, CubeEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, CubeIndicesSize * sizeof(float), CubeIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// Attractor
	glGenVertexArrays(1, &AttractorVAO);
	glGenBuffers(1, &AttractorVBO);

	glBindVertexArray(AttractorVAO);

	glBindBuffer(GL_ARRAY_BUFFER, AttractorVBO);
	glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float), &AttractorLocation, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	cudaGraphicsGLRegisterBuffer(&CUDALocationVBO, LocationVBO, cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer(&CUDAColorVBO, ColorVBO, cudaGraphicsMapFlagsNone);
}

void GParticleSystem::Display(glm::mat4 Projection, glm::mat4 View, glm::vec2 LightDirection, glm::vec3 ViewDirection, float PointScale)
{
	glBindVertexArray(VAO);
	Shader.Use();
	Shader.SetVec2r("ULightDirection", LightDirection);
	Shader.SetVec3("UViewDirection", ViewDirection);
	Shader.SetProjectionViewModel(Projection, View, glm::mat4(1.f));
	Shader.Set1f("UPointRadius", Radius);
	Shader.Set1f("UPointScale", PointScale);
	glDrawArrays(GL_POINTS, 0, Number);

	if (bUseBoundaries)
	{
		glBindVertexArray(CubeVAO);
		CubeShader.Use();
		CubeShader.SetProjectionViewModel(Projection, View, glm::scale(glm::mat4(1.f), glm::vec3(BoundarySize)));
		glDrawElements(GL_LINES, CubeIndicesSize, GL_UNSIGNED_INT, 0);
	}

	if (bUseAttractor)
	{
		glBindVertexArray(AttractorVAO);
		AttractorShader.Use();
		AttractorShader.SetVec3("ULocation", glm::vec3(AttractorLocation.x, AttractorLocation.y, AttractorLocation.z));
		AttractorShader.SetVec2r("ULightDirection", LightDirection);
		AttractorShader.SetVec3("UViewDirection", ViewDirection);
		AttractorShader.SetProjectionViewModel(Projection, View, glm::mat4(1.f));
		AttractorShader.Set1f("UPointSize", AttractorGravity);
		AttractorShader.Set1f("UPointScale", PointScale);
		glDrawArrays(GL_POINTS, 0, 1);
	}
}

void GParticleSystem::Update(bool bUseCPU)
{
	if (bUseCPU)
	{
		glBindBuffer(GL_ARRAY_BUFFER, LocationVBO);
		float* LocationData = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
		//glBindBuffer(GL_ARRAY_BUFFER, ColorVBO);
		//float* ColorData = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

		if (bResetLocation)
		{
			bResetLocation = false;
			RandomCUDAArrayCPU(LocationData, -BoundarySize / 2.f + Radius, BoundarySize / 2.f - Radius, 3 * Number);
			SetCUDAArrayCPU((float3 *)Velocity, make_float3(0.f, 0.f, 0.f), Number);
		}

		if (bUseGravity)
		{
			SetCUDAArrayCPU((float3 *)Acceleration, Gravity, Number);
		}
		else
		{
			SetCUDAArrayCPU((float3 *)Acceleration, make_float3(0.f, 0.f, 0.f), Number);
		}

		if (bUseAttractor)
		{
			ApplyAttractorCPU((float3 *)LocationData, (float3 *)Acceleration, AttractorLocation, AttractorGravity, Number);
		}

		if (bUseCollisions)
		{
			glm::vec3 GLMGravityDirection = glm::normalize(glm::vec3(Gravity.x, Gravity.y, Gravity.z));
			float3 GravityDirection = PositiveFloat3(GLMGravityDirection);
			CheckCollisionsCPU((float3 *)LocationData, (float3 *)Velocity, (float3 *)Acceleration, Number, Radius, CollisionsSpring, CollisionsDamping, CollisionsShear, CollisionsAttraction);
		}

		UpdateLocationCPU(LocationData, Velocity, Acceleration, 3 * Number);

		if (bUseBoundaries)
		{
			CheckBoundariesCPU((float3 *)LocationData, (float3 *)Velocity, Number, Radius, BoundarySize / 2.f, BoundaryFriction);
		}

		//CheckLifespan((float3 *)LocationData, (float3 *)Velocity, Lifespan, (float4 *)ColorData, Number);

		glBindBuffer(GL_ARRAY_BUFFER, LocationVBO);
		glUnmapBuffer(GL_ARRAY_BUFFER);
		//glBindBuffer(GL_ARRAY_BUFFER, ColorVBO);
		//glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else
	{
		// Map OpenGL buffer object for writing from CUDA
		float *LocationData;
		//float *ColorData;
		size_t Bytes;

		checkCudaErrors(cudaGraphicsMapResources(1, &CUDALocationVBO, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&LocationData, &Bytes,
			CUDALocationVBO));

		//checkCudaErrors(cudaGraphicsMapResources(1, &CUDAColorVBO, 0));
		//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ColorData, &Bytes,
		//	CUDAColorVBO));

		if (bResetLocation)
		{
			bResetLocation = false;
			RandomCUDAArray(LocationData, -BoundarySize / 2.f + Radius, BoundarySize / 2.f - Radius, 3 * Number);
			SetCUDAArray((float3 *)Velocity, make_float3(0.f, 0.f, 0.f), Number);
		}

		if (bUseGravity)
		{
			SetCUDAArray((float3 *)Acceleration, Gravity, Number);
		}
		else
		{
			SetCUDAArray((float3 *)Acceleration, make_float3(0.f, 0.f, 0.f), Number);
		}

		if (bUseAttractor)
		{
			ApplyAttractor((float3 *)LocationData, (float3 *)Acceleration, AttractorLocation, AttractorGravity, Number);
		}

		if (bUseCollisions)
		{
			glm::vec3 GLMGravityDirection = glm::normalize(glm::vec3(Gravity.x, Gravity.y, Gravity.z));
			float3 GravityDirection = PositiveFloat3(GLMGravityDirection);
			CheckCollisions((float3 *)LocationData, (float3 *)Velocity, (float3 *)Acceleration, Number, Radius, CollisionsSpring, CollisionsDamping, CollisionsShear, CollisionsAttraction);
		}

		UpdateLocation(LocationData, Velocity, Acceleration, 3 * Number);

		if (bUseBoundaries)
		{
			CheckBoundaries((float3 *)LocationData, (float3 *)Velocity, Number, Radius, BoundarySize / 2.f, BoundaryFriction);
		}

		//CheckLifespan((float3 *)LocationData, (float3 *)Velocity, Lifespan, (float4 *)ColorData, Number);

		checkCudaErrors(cudaGraphicsUnmapResources(1, &CUDALocationVBO, 0));
		//checkCudaErrors(cudaGraphicsUnmapResources(1, &CUDAColorVBO, 0));
	}
}

float* GParticleSystem::GetFirstLocation()
{
	return Location;
}

void GParticleSystem::SetRandomVelocity(glm::vec3 Minimum, glm::vec3 Maximium)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		Velocity[i * 3] = RandFloat(Minimum.x, Maximium.x);
		Velocity[i * 3 + 1] = RandFloat(Minimum.y, Maximium.y);
		Velocity[i * 3 + 2] = RandFloat(Minimum.z, Maximium.z);
	}
}

void GParticleSystem::SetLocation(glm::vec3 InLocation)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		Location[i * 3] = InLocation.x;
		Location[i * 3 + 1] = InLocation.y;
		Location[i * 3 + 2] = InLocation.z;
	}
}

void GParticleSystem::Reset()
{
	glBindBuffer(GL_ARRAY_BUFFER, LocationVBO);
	float* LocationData = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for (unsigned int i = 0; i < 3 * Number; ++i)
	{
		LocationData[i] = Location[i];
	}

	glBindBuffer(GL_ARRAY_BUFFER, LocationVBO);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	SetCUDAArray((float3 *)Velocity, make_float3(0.f, 0.f, 0.f), Number);
	SetCUDAArray((float3 *)Acceleration, make_float3(0.f, 0.f, 0.f), Number);
}

void GParticleSystem::UpdateAttractor(bool InbUseAttractor, glm::vec3 InAttractorLocation, float InAttractorGravity)
{
	bUseAttractor = InbUseAttractor;
	AttractorLocation = make_float3(InAttractorLocation.x, InAttractorLocation.y, InAttractorLocation.z);
	AttractorGravity = InAttractorGravity;
}

void GParticleSystem::ResetLocation()
{
	bResetLocation = true;
}

void GParticleSystem::UpdateBoundaries(bool InbUseBoundaries, float InBoundarySize, float InBoundaryFriction)
{
	bUseBoundaries = InbUseBoundaries;
	BoundarySize = InBoundarySize;
	BoundaryFriction = InBoundaryFriction;
}

void GParticleSystem::UpdateGravity(bool InbUseGravity, glm::vec3 InGravity)
{
	bUseGravity = InbUseGravity;
	Gravity = make_float3(InGravity.x, InGravity.y, InGravity.z);
}

void GParticleSystem::UpdateCollisions(bool InbUseCollisions, float InCollisionsSpring, float InCollisionsDamping, float InCollisionsShear, float InCollisionsAttraction)
{
	bUseCollisions = InbUseCollisions;
	CollisionsSpring = InCollisionsSpring;
	CollisionsDamping = InCollisionsDamping;
	CollisionsShear = InCollisionsShear;
	CollisionsAttraction = InCollisionsAttraction;
}

void GParticleSystem::SetCUDAArrayCPU(float3* Coordinates, float3 Vector, unsigned int Number)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		Coordinates[i] = Vector;
	}
}

void GParticleSystem::RandomCUDAArrayCPU(float* Array, float Minimum, float Maximum, unsigned int Number)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		Array[i] = RandFloat(Minimum, Maximum);
	}
}

void GParticleSystem::ApplyAttractorCPU(float3* Location, float3* Acceleration, float3 AttractorLocation, float AttractorGravity, unsigned int Number)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		float3 Force = AttractorLocation - Location[i];
		float Distance = Clamp(Length(Force), 1.f, 5.f);

		Force = Normalize(Force);
		float Strength = AttractorGravity / (Distance * Distance);
		Force = Force * Strength;
		Acceleration[i] += Force;
	}
}

void GParticleSystem::CheckCollisionsCPU(float3* Location, float3* Velocity, float3* Acceleration, unsigned int Number, float Radius, float Spring, float Damping, float Shear, float Attraction)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		for (unsigned int j = 0; j < Number; ++j)
		{
			if (i != j)
			{
				float3 RelativeLocation = Location[j] - Location[i];
				float Distance = Length(RelativeLocation);
				if (Distance < 2 * Radius)
				{
					float3 Direction = RelativeLocation / Distance;
					float3 RelativeVelocity = Velocity[j] - Velocity[i];
					float3 TangencialVelocity = RelativeVelocity - Dot(RelativeVelocity, Direction) * Direction;

					Acceleration[i] -= Spring * (2 * Radius - Distance) * Direction;
					Acceleration[i] += Damping * RelativeVelocity;
					Acceleration[i] += Shear * TangencialVelocity;
					Acceleration[i] += Attraction * RelativeLocation;
				}
			}
		}
	}
}

void GParticleSystem::UpdateLocationCPU(float* Location, float* Velocity, float* Acceleration, unsigned int Number)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		Velocity[i] += Acceleration[i];
		Location[i] += Velocity[i];
	}
}

void GParticleSystem::CheckBoundariesCPU(float3* Location, float3* Velocity, unsigned int Number, float Radius, float Limit, float Friction)
{
	for (unsigned int i = 0; i < Number; ++i)
	{
		if (Location[i].x < -Limit + Radius)
		{
			Location[i].x = -Limit + Radius;
			Velocity[i].x *= -(1.f - Friction);
			Velocity[i].y *= (1.f - Friction);
			Velocity[i].z *= (1.f - Friction);
		}
		if (Location[i].x > Limit - Radius)
		{
			Location[i].x = Limit - Radius;
			Velocity[i].x *= -(1.f - Friction);
			Velocity[i].y *= (1.f - Friction);
			Velocity[i].z *= (1.f - Friction);
		}
		if (Location[i].y < -Limit + Radius)
		{
			Location[i].y = -Limit + Radius;
			Velocity[i].x *= (1.f - Friction);
			Velocity[i].y *= -(1.f - Friction);
			Velocity[i].z *= (1.f - Friction);
		}
		if (Location[i].y > Limit - Radius)
		{
			Location[i].y = Limit - Radius;
			Velocity[i].x *= (1.f - Friction);
			Velocity[i].y *= -(1.f - Friction);
			Velocity[i].z *= (1.f - Friction);
		}
		if (Location[i].z < -Limit + Radius)
		{
			Location[i].z = -Limit + Radius;
			Velocity[i].x *= (1.f - Friction);
			Velocity[i].y *= (1.f - Friction);
			Velocity[i].z *= -(1.f - Friction);
		}
		if (Location[i].z > Limit - Radius)
		{
			Location[i].z = Limit - Radius;
			Velocity[i].x *= (1.f - Friction);
			Velocity[i].y *= (1.f - Friction);
			Velocity[i].z *= -(1.f - Friction);
		}
	}
}