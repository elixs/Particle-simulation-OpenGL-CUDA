#include "Main.h"

#include <stdio.h>

#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <utility>
#include <limits>
#include <cmath>

// glm
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// imgui
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include "CUDA_helper/helper_cuda.h"

#include "Shader.h"
#include "Utils.h"

#include "ParticleSystem.h"

#include "Main.cuh"

#include "Defines.h"

#include "resource.h"

#ifdef PERFORMANCE_TEST
#include <chrono>

unsigned int Iterations = 0;
#endif // PERFORMANCE_TEST

void CUDAMain(cudaGraphicsResource_t* CUDASphereVBO, float Time)
{
	// map OpenGL buffer object for writing from CUDA
	float3 *Data;
	checkCudaErrors(cudaGraphicsMapResources(1, CUDASphereVBO, 0));
	size_t Bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&Data, &Bytes, *CUDASphereVBO));

	//printf("CUDA mapped VBO: May access %zd bytes\n", Bytes);

	// execute the kernel

	SphereWave(Data, TEST_SEGMENTS, TEST_RINGS, TEST_RADIUS, Time);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, CUDASphereVBO, 0));
}

#ifdef _CONSOLE
int main()
#else _WINDOWS
int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#endif
{
	GLFWwindow* Window;
	Init(Window, u8"(~� - �)~");

	GShader SphereShader;
	GShader ArrowShader;
	GShader PointLightShader;

	unsigned int Particles;

#ifdef STAND_ALONE
	// To create stand-alone .exe
	SphereShader = GShader(DefaultVert, DefaultFrag);
	ArrowShader = GShader(ArrowVert, ArrowFrag);
	PointLightShader = GShader(PointLightVert, PointLightFrag);

	Particles = 0;
#else
	// Shaders
	SphereShader = GShader("Shaders/Default.vert", "Shaders/Default.frag");
	ArrowShader = GShader("Shaders/Arrow.vert", "Shaders/Arrow.frag");
	PointLightShader = GShader("Shaders/PointLight.vert", "Shaders/PointLight.frag");

	Particles = 1000;
#endif // STAND_ALONE

	GParticleSystem ParticleSystem(Particles);
	ParticleSystem.SetRandomVelocity(glm::vec3(-0.003f, 0.f, -0.003f), glm::vec3(0.003f, 0.006f, 0.003f));

	ParticleSystem.InitGLCUDA();

	// Test
	unsigned int SphereVAO, SphereVBO, SphereEBO;
	int SphereIndicesSize;
	SphereInit(SphereVAO, SphereVBO, SphereEBO, SphereIndicesSize, TEST_SEGMENTS, TEST_RINGS, TEST_RADIUS, GenerateSphere);

	//// Directional Light Arrow
	//  Cylinder
	unsigned int CylinderVAO, CylinderVBO, CylinderEBO;
	int CylinderIndicesSize;
	ArrowInit(CylinderVAO, CylinderVBO, CylinderEBO, CylinderIndicesSize, 16, 0.01f, 0.2f, GenerateCylinder);

	// Cone
	unsigned int ConeVAO, ConeVBO, ConeEBO;
	int ConeIndicesSize;
	ArrowInit(ConeVAO, ConeVBO, ConeEBO, ConeIndicesSize, 16, 0.02f, 0.04f, GenerateCone);

	// PointLight
	unsigned int PointLightVAO, PointLightVBO, PointLightEBO;
	int PointLightIndicesSize;
	PointLightInit(PointLightVAO, PointLightVBO, PointLightEBO, PointLightIndicesSize, 32, 16, 1.f, GenerateSphere);

	cudaGraphicsResource_t CUDASphereVBO;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&CUDASphereVBO, SphereVBO, cudaGraphicsMapFlagsNone));

	SphereShaderInit(SphereShader);

	ArrowShaderInit(ArrowShader);

	//// ImGui variables
	ImVec4 ClearColor = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	// Direction Light
	bool bUseDirectionalLight = true;
	glm::vec2 DLDirection(270.f, 315.f);
	glm::vec2 LastDLDirection = DLDirection;
	float LastCameraYaw = Camera.Yaw;
	float LastCameraPitch = Camera.Pitch;
	glm::vec3 DLAmbient(0.5f);
	glm::vec3 DLDiffuse(0.5f);
	glm::vec3 DLSpectular(0.5f);

	// Point Light
	bool bUsePointLight = true;
	glm::vec3 PLLocation(0.f, 7.f, 10.f);
	glm::vec3 PLAmbient(0.25f);
	glm::vec3 PLDiffuse(0.75f);
	glm::vec3 PLSpectular(1.f);
	float PLConstant = 1.f;
	float PLLinear = 0.022f;
	float PLQuadratic = 0.0019f;

	// Spot Light
	bool bUseSpotLight = true;
	glm::vec3 SLAmbient(0.f);
	glm::vec3 SLDiffuse(1.f);
	glm::vec3 SLSpectular(1.f);
	float SLConstant = 1.f;
	float SLLinear = 0.09f;
	float SLQuadratic = 0.032f;
	float SLCutOff = 12.5f;
	float SLOuterCutOff = 15.f;

	float CameraSpeed = Camera.MovementSpeed;

	// Particle Simulation
	bool bUseBoundaries = true;
	float BoundarySize = BOX_SIZE;
	float BoundaryFriction = 0.2f;

	bool bUseGravity = true;
	glm::vec3 Gravity(0.f, -0.001f, 0.f);

	bool bUseAttractor = false;
	glm::vec3 AttractorLocation(0.f);
	float AttractorGravity = 0.001f;

	bool bUseCollisions = true;
	float CollisionsSpring = 0.5f;
	float CollisionsDamping = 0.03f;
	float CollisionsShear = 0.1f;
	float CollisionsAttraction = 0.f;

	CUDAMain(&CUDASphereVBO, SphereTime);

#ifdef PERFORMANCE_TEST
	bUseAttractor = true;
	AttractorGravity = 0.0025f;
	BoundarySize = 7.f;

	unsigned int Iterations = 0;

	auto Begin = std::chrono::steady_clock::now();
#endif // PERFORMANCE_TEST

	while (!glfwWindowShouldClose(Window))
	{
#ifdef PERFORMANCE_TEST
		++Iterations;

		if (Iterations == 100)
		{
			bUseGravity = false;
		}
		if (Iterations == 200)
		{
			AttractorGravity *= -1;
		}
		if (Iterations == 300)
		{
			CollisionsAttraction = 0.1f;
			AttractorGravity *= -1;
		}
		if (Iterations == 400)
		{
			bUseGravity = true;
			bUseAttractor = false;
		}
		if (Iterations == 500)
		{
			bUseCollisions = false;
			bUseAttractor = true;
			AttractorGravity *= 2;
			bUseBoundaries = false;
		}
		if (Iterations == 600)
		{
			bUseBoundaries = true;
			bUseCollisions = true;
			bUseAttractor = true;
			AttractorGravity *= -1;
		}
		if (Iterations == 700)
		{
			Gravity.x = Gravity.y;
			Gravity.z = -Gravity.y;
			Gravity.y = 0.f;
			bUseCollisions = true;
			bUseAttractor = true;
			AttractorGravity *= -1;
		}
		if (Iterations == 800)
		{
			Gravity.x *= -1;
			Gravity.z *= -1;
			AttractorGravity *= -2.5f;
		}
		if (Iterations == 900)
		{
			Gravity.y = Gravity.x;
			Gravity.x = 0.f;
			Gravity.z = 0.f;
			AttractorGravity *= -0.2f;
		}
		if (Iterations == 1000)
		{
			auto End = std::chrono::steady_clock::now();
			auto Time = 1.0 * std::chrono::duration_cast<std::chrono::nanoseconds>(End - Begin).count() / 1000000000.0;

			ParticleSystem.Reset();

			Gravity.y = -0.001f;
			CollisionsAttraction = 0.f;

			if (bUseCPU)
			{
				std::cout << "CPU: " << Time << " seconds" << std::endl << std::endl;
				bUseCPU = false;
				bPause = true;
			}
			else
			{
				std::cout << "GPU: " << Time << " seconds" << std::endl << std::endl;

				Iterations = 0;
				bUseCPU = true;
				Begin = std::chrono::steady_clock::now();
			}
		}
#endif // PERFORMANCE_TEST

		int Width, Height;
		glfwGetFramebufferSize(Window, &Width, &Height);

		ImGuiUpdate();

		if (bShowHelp)
		{
			ShowHelp(Width);
		}

		if (CurrentState == EState::OnMenu)
		{
			// Main GUI
			ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
			if (bSphereTest)
			{
				ImGui::SetNextWindowSize(ImVec2(500, 725.f));
			}
			else if (bFixDirectionalLight)
			{
				ImGui::SetNextWindowSize(ImVec2(500, 585.f));
			}
			else
			{
				ImGui::SetNextWindowSize(ImVec2(500, 610.f));
			}
			ImGui::Begin(u8"(/� - �)/");                          // Create a window called "Hello, world!" and append into it.
			ImGui::ColorEdit3("Clear Color", (float*)&ClearColor); // Edit 3 floats representing a color
			ImGui::DragFloat("Camera Speed", &CameraSpeed, 0.1f);
			ImGui::Checkbox("Pause", &bPause);
			ImGui::Checkbox("Sphere Test", &bSphereTest);
			if (bSphereTest)
			{
				ImGui::Checkbox("Wireframe", &bWireframe);
			}
			else
			{
				ImGui::Checkbox("Use CPU", &bUseCPU);

				if (!ImGui::CollapsingHeader("Boundaries"))
				{
					ImGui::PushID(3);
					ImGui::Checkbox("Active", &bUseBoundaries);
					ImGui::DragFloat("Size", &BoundarySize, 0.1f, 0.2f, std::numeric_limits<float>::max());
					ImGui::DragFloat("Friction", &BoundaryFriction, 0.001f, 0.0f, 1.0);
					ImGui::PopID();
				}

				if (!ImGui::CollapsingHeader("Gravity"))
				{
					ImGui::PushID(4);
					ImGui::Checkbox("Active", &bUseGravity);
					ImGui::DragFloat3("Force", (float*)&Gravity, 0.000001f, 0.f, 0.f, "%.6f");
					ImGui::PopID();
				}

				if (!ImGui::CollapsingHeader("Attractor"))
				{
					ImGui::PushID(5);
					ImGui::Checkbox("Active", &bUseAttractor);
					ImGui::DragFloat3("Location", (float*)&AttractorLocation, 0.1f);
					ImGui::DragFloat("Force", &AttractorGravity, 0.0001f, 0.f, 0.f, "%.4f");
					ImGui::PopID();
				}
				if (!ImGui::CollapsingHeader("Collisions"))
				{
					ImGui::PushID(6);
					ImGui::Checkbox("Active", &bUseCollisions);
					ImGui::DragFloat("Spring", &CollisionsSpring, 0.001f, 0.f, 1.f);
					ImGui::DragFloat("Damping", &CollisionsDamping, 0.001f, 0.0f, 0.1f);
					ImGui::DragFloat("Shear", &CollisionsShear, 0.001f, 0.f, 0.1f);
					ImGui::DragFloat("Attraction", &CollisionsAttraction, 0.001f, 0.f, 0.1f);
					ImGui::PopID();
				}
			}
			if (!ImGui::CollapsingHeader("Directional Light"))
			{
				ImGui::PushID(0);
				if (bSphereTest)
				{
					ImGui::Checkbox("Active", &bUseDirectionalLight);
				}
				else
				{
					ImGui::Checkbox("Fix", &bFixDirectionalLight);
				}
				if (!bFixDirectionalLight || bSphereTest)
				{
					SliderRotation("Direction", (float*)&DLDirection);
				}
				if (bSphereTest)
				{
					ImGui::ColorEdit3("Ambient", (float*)&DLAmbient);
					ImGui::ColorEdit3("Diffuse", (float*)&DLDiffuse);
					ImGui::ColorEdit3("Specular", (float*)&DLSpectular);
				}
				ImGui::PopID();
			}
			if (bSphereTest)
			{
				if (!ImGui::CollapsingHeader("Point Light"))
				{
					ImGui::PushID(1);
					ImGui::Checkbox("Active", &bUsePointLight);
					ImGui::DragFloat3("Location", (float*)&PLLocation, 0.1f);
					ImGui::ColorEdit3("Ambient", (float*)&PLAmbient);
					ImGui::ColorEdit3("Diffuse", (float*)&PLDiffuse);
					ImGui::ColorEdit3("Specular", (float*)&PLSpectular);
					ImGui::DragFloat("Constant", &PLConstant, 0.01f);
					ImGui::DragFloat("Linear", &PLLinear, 0.001f);
					ImGui::DragFloat("Quadratic", &PLQuadratic, 0.0001f, 0.f, 0.f, "%.4f");
					ImGui::PopID();
				}
				if (!ImGui::CollapsingHeader("Spot Light"))
				{
					ImGui::PushID(2);
					ImGui::Checkbox("Active", &bUseSpotLight);
					ImGui::ColorEdit3("Ambient", (float*)&SLAmbient);
					ImGui::ColorEdit3("Diffuse", (float*)&SLDiffuse);
					ImGui::ColorEdit3("Specular", (float*)&SLSpectular);
					ImGui::DragFloat("Constant", &SLConstant, 0.01f);
					ImGui::DragFloat("Linear", &SLLinear, 0.001f);
					ImGui::DragFloat("Quadratic", &SLQuadratic, 0.0001f, 0.f, 0.f, "%.4f");
					ImGui::DragFloat("Cut Off", &SLCutOff, 0.1f);
					ImGui::DragFloat("Outer Cut Off", &SLOuterCutOff, 0.1f);
					ImGui::PopID();
				}
			}
			ImGui::End();
		}

		float CurrentFrame = (float)glfwGetTime();
		DeltaTime = CurrentFrame - LastFrame;
		LastFrame = CurrentFrame;

		Camera.MovementSpeed = CameraSpeed;

		if (bSphereTest && !bPause)
		{
			SphereTime += DeltaTime;
			CUDAMain(&CUDASphereVBO, SphereTime);
		}

		ProcessInput(Window);

		glClearColor(ClearColor.x, ClearColor.y, ClearColor.z, ClearColor.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 Projection = glm::perspective(glm::radians(Camera.Zoom), (float)Width / (float)Height, 0.1f, 100.f);
		glm::mat4 View = Camera.GetViewMatrix();
		glm::mat4 Model(1.f);

		if (bSphereTest)
		{
			glBindVertexArray(SphereVAO);
			SphereShader.Use();

			//// Lights
			// Directional Light
			if (bUseDirectionalLight)
			{
				SphereShader.SetDirectionalLight(-1, DLAmbient, DLDiffuse, DLSpectular, DLDirection);
			}
			else
			{
				SphereShader.SetDirectionalLight(-1, glm::vec3(0.f), glm::vec3(0.f), glm::vec3(0.f), DLDirection);
			}

			// Point Light
			if (bUsePointLight)
			{
				SphereShader.SetPointLight(0, PLAmbient, PLDiffuse, PLSpectular, PLLocation, PLConstant, PLLinear, PLQuadratic);
			}
			else
			{
				SphereShader.SetPointLight(0, glm::vec3(0.f), glm::vec3(0.f), glm::vec3(0.f), PLLocation, PLConstant, PLLinear, PLQuadratic);
			}

			// Spot Light
			if (bUseSpotLight)
			{
				SphereShader.SetSpotLight(-1, SLAmbient, SLDiffuse, SLSpectular, Camera.Location, Camera.Front, SLConstant, SLLinear, SLQuadratic, glm::cos(glm::radians(SLCutOff)), glm::cos(glm::radians(SLOuterCutOff)));
			}
			else
			{
				SphereShader.SetSpotLight(-1, glm::vec3(0.f), glm::vec3(0.f), glm::vec3(0.f), Camera.Location, Camera.Front, SLConstant, SLLinear, SLQuadratic, glm::cos(glm::radians(SLCutOff)), glm::cos(glm::radians(SLOuterCutOff)));
			}

			SphereShader.SetVec3("UViewLocation", Camera.Location);

			SphereShader.SetMat4("UProjection", Projection);
			SphereShader.SetMat4("UView", View);
			SphereShader.SetMat4("UModel", Model);

			if (bWireframe)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			glDrawElements(GL_TRIANGLES, SphereIndicesSize, GL_UNSIGNED_INT, 0);

			if (bUsePointLight)
			{
				PointLightShader.Use();

				PointLightShader.SetVec3("UViewLocation", Camera.Location);

				PointLightShader.SetMat4("UProjection", Projection);
				PointLightShader.SetMat4("UView", View);
				Model = glm::mat4(1.f);
				Model = glm::translate(Model, PLLocation);
				PointLightShader.SetMat4("UModel", Model);

				glBindVertexArray(PointLightVAO);
				glDrawElements(GL_TRIANGLES, PointLightIndicesSize, GL_UNSIGNED_INT, 0);
			}

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
		else
		{
			if (bReset)
			{
				bReset = false;
				ParticleSystem.ResetLocation();
			}

			if (bFixDirectionalLight)
			{
				DLDirection = LastDLDirection + glm::vec2(Camera.Yaw - LastCameraYaw, Camera.Pitch - LastCameraPitch);
				DLDirection.x = std::fmodf(DLDirection.x + 360.f, 360);
				DLDirection.y = std::fmodf(DLDirection.y + 360.f, 360);
			}
			else
			{
				LastDLDirection = DLDirection;
				LastCameraYaw = Camera.Yaw;
				LastCameraPitch = Camera.Pitch;
			}

			ParticleSystem.UpdateCollisions(bUseCollisions, CollisionsSpring, CollisionsDamping, CollisionsShear, CollisionsAttraction);
			ParticleSystem.UpdateBoundaries(bUseBoundaries, BoundarySize, BoundaryFriction);
			ParticleSystem.UpdateGravity(bUseGravity, Gravity);
			ParticleSystem.UpdateAttractor(bUseAttractor, AttractorLocation, AttractorGravity);

			if (!bPause)
			{
				ParticleSystem.Update(bUseCPU);
			}

			ParticleSystem.Display(Projection, View, DLDirection, Camera.Front, (float)Height / glm::tan(1.f / 3.f * glm::radians(Camera.Zoom)));
		}

		glClear(GL_DEPTH_BUFFER_BIT);

		if (bUseDirectionalLight)
		{
			ArrowShader.Use();

			ArrowShader.SetVec3("USpotLight.Location", Camera.Location);
			ArrowShader.SetVec3("USpotLight.Direction", 2.f * Camera.Front + glm::vec3(0.f, 0.5f, 0.f));

			ArrowShader.SetVec3("UViewLocation", Camera.Location);

			ArrowShader.SetMat4("UProjection", Projection);
			ArrowShader.SetMat4("UView", View);

			Model = glm::mat4(1.f);
			if (bSphereTest || !bUseGravity)
			{
				Model = glm::translate(Model, Camera.Location + 2.f * Camera.Front + 0.5f * Camera.Up);
			}
			else
			{
				Model = glm::translate(Model, Camera.Location + 2.f * Camera.Front + 0.5f * Camera.Up + 0.5f * Camera.Right);
			}

			glm::vec3 Direction = -GetNormal(DLDirection);
			glm::vec3 Front = Direction;
			glm::vec3 Right = glm::normalize(glm::cross(Front, glm::vec3(0.f, 1.f, 1.f)));
			glm::vec3 Up = glm::normalize(glm::cross(Right, Front));

			glm::mat4 Rotation = glm::inverse(glm::lookAt(glm::vec3(0.f, 0.f, 0.f), glm::vec3(Direction.x, Direction.y, Direction.z), Up));

			Model = Model * Rotation;
			Model = glm::rotate(Model, glm::radians(270.f), glm::vec3(0.f, 1.f, 0.f));
			Model = glm::rotate(Model, glm::radians(270.f), glm::vec3(0.f, 0.f, 1.f));

			ArrowShader.SetMat4("UModel", Model);
			glBindVertexArray(CylinderVAO);
			glDrawElements(GL_TRIANGLES, CylinderIndicesSize, GL_UNSIGNED_INT, 0);

			Model = glm::translate(Model, glm::vec3(0.f, 0.1f, 0.f));
			ArrowShader.SetMat4("UModel", Model);
			glBindVertexArray(ConeVAO);
			glDrawElements(GL_TRIANGLES, ConeIndicesSize, GL_UNSIGNED_INT, 0);
		}

		if (bUseGravity && !bSphereTest)
		{
			ArrowShader.Use();

			ArrowShader.SetVec3("USpotLight.Location", Camera.Location);
			ArrowShader.SetVec3("USpotLight.Direction", 2.f * Camera.Front + glm::vec3(0.f, 0.5f, 0.f));

			ArrowShader.SetVec3("UViewLocation", Camera.Location);

			ArrowShader.SetMat4("UProjection", Projection);
			ArrowShader.SetMat4("UView", View);

			Model = glm::mat4(1.f);
			Model = glm::translate(Model, Camera.Location + 2.f * Camera.Front + 0.5f * Camera.Up - 0.5f * Camera.Right);

			glm::vec3 Direction = -glm::normalize(Gravity);
			glm::vec3 Front = Direction;
			glm::vec3 Right = glm::normalize(glm::cross(Front, glm::vec3(0.f, 1.f, 1.f)));
			glm::vec3 Up = glm::normalize(glm::cross(Right, Front));

			glm::mat4 Rotation = glm::inverse(glm::lookAt(glm::vec3(0.f, 0.f, 0.f), glm::vec3(Direction.x, Direction.y, Direction.z), Up));

			Model = Model * Rotation;
			Model = glm::rotate(Model, glm::radians(270.f), glm::vec3(0.f, 1.f, 0.f));
			Model = glm::rotate(Model, glm::radians(270.f), glm::vec3(0.f, 0.f, 1.f));

			ArrowShader.SetMat4("UModel", Model);
			glBindVertexArray(CylinderVAO);
			glDrawElements(GL_TRIANGLES, CylinderIndicesSize, GL_UNSIGNED_INT, 0);

			Model = glm::translate(Model, glm::vec3(0.f, 0.1f, 0.f));
			ArrowShader.SetMat4("UModel", Model);
			glBindVertexArray(ConeVAO);
			glDrawElements(GL_TRIANGLES, ConeIndicesSize, GL_UNSIGNED_INT, 0);
		}

		// Rendering
		ImGui::Render();

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(Window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1, &SphereVAO);
	glDeleteBuffers(1, &SphereVBO);
	glDeleteBuffers(1, &SphereEBO);

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(Window);
	glfwTerminate();
	return 0;
}

void Init(GLFWwindow* &Window, const char* Title)
{
	checkCudaErrors(cudaSetDevice(0));

	glfwSetErrorCallback(ErrorCallback);
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	Window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, Title, NULL, NULL);
	if (Window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}
	glfwMakeContextCurrent(Window);

	// Callbacks
	glfwSetFramebufferSizeCallback(Window, FramebufferSizeCallback);
	glfwSetKeyCallback(Window, KeyCallback);
	glfwSetCharCallback(Window, CharacterCallback);
	glfwSetCursorPosCallback(Window, MouseCallback);
	glfwSetScrollCallback(Window, ScrollCallback);

	glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	io.IniFilename = "imgui/imgui.ini";

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(Window, false);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	// Setup Style
	ImGui::StyleColorsClassic();

	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_PROGRAM_POINT_SIZE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void SphereInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &SphereIndicesSize, int Segments, int Rings, float Radius, void(*Generate)(float*&, int*&, int, int, float, int&, int&))
{
	// in
	float* Sphere;
	int SphereSize;

	// out
	int* SphereIndices;
	//int SphereIndicesSize;

	Generate(Sphere, SphereIndices, Segments, Rings, Radius, SphereSize, SphereIndicesSize);

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, SphereSize * sizeof(float), Sphere, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, SphereIndicesSize * sizeof(float), SphereIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

void ArrowInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &ArrowIndicesSize, int Vertices, float Radius, float Legth, void(*Generate)(float*&, int*&, int, float, float, int&, int&, bool))
{
	// in
	float* Arrow;
	int ArrowSize;

	// out
	int* ArrowIndices;
	//int ArrowIndicesSize;

	Generate(Arrow, ArrowIndices, Vertices, Radius, Legth, ArrowSize, ArrowIndicesSize, true);

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, ArrowSize * sizeof(float), Arrow, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, ArrowIndicesSize * sizeof(float), ArrowIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

void PointLightInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &PointLightIndicesSize, int Segments, int Rings, float Radius, void(*Generate)(float*&, int*&, int, int, float, int&, int&))
{
	// in
	float* PointLight;
	int PointLightSize;

	// out
	int* PointLightIndices;
	//int PointLightIndicesSize;

	Generate(PointLight, PointLightIndices, Segments, Rings, Radius, PointLightSize, PointLightIndicesSize);

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, PointLightSize * sizeof(float), PointLight, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, PointLightIndicesSize * sizeof(float), PointLightIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

void CubeInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &CubeIndicesSize, float Length, void(*Generate)(float*&, int*&, float, int&, int&))
{
	// in
	float* Cube;
	int CubeSize;

	// out
	int* CubeIndices;
	//int CubeIndicesSize;

	Generate(Cube, CubeIndices, Length, CubeSize, CubeIndicesSize);

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, CubeSize * sizeof(float), Cube, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, CubeIndicesSize * sizeof(float), CubeIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
}

void SphereShaderInit(GShader &SphereShader)
{
	SphereShader.Use();

	SphereShader.Set3f("UMaterial.Ambient", 0.1745f, 0.01175f, 0.01175f);
	SphereShader.Set3f("UMaterial.Diffuse", 0.61424f, 0.04136f, 0.04136f);
	SphereShader.Set3f("UMaterial.Specular", 0.727811f, 0.626959f, 0.626959f);
	SphereShader.Set1f("UMaterial.Shininess", 76.8f);

	SphereShader.SetVec3("USpotLight.Light.Ambient", glm::vec3(1.f));
	SphereShader.SetVec3("USpotLight.Light.Diffuse", glm::vec3(1.f));
	SphereShader.SetVec3("USpotLight.Light.Specular", glm::vec3(1.f));
	SphereShader.Set1f("USpotLight.Constant", 1.f);
	SphereShader.Set1f("USpotLight.Linear", 0.09f);
	SphereShader.Set1f("USpotLight.Quadratic", 0.032f);
	SphereShader.Set1f("USpotLight.CutOff", glm::cos(glm::radians(12.5f)));
	SphereShader.Set1f("USpotLight.OuterCutOff", glm::cos(glm::radians(15.f)));
}

void ArrowShaderInit(GShader &ArrowShader)
{
	ArrowShader.Use();

	ArrowShader.Set3f("UMaterial.Ambient", 0.1745f, 0.01175f, 0.01175f);
	ArrowShader.Set3f("UMaterial.Diffuse", 0.61424f, 0.04136f, 0.04136f);
	ArrowShader.Set3f("UMaterial.Specular", 0.727811f, 0.626959f, 0.626959f);
	ArrowShader.Set1f("UMaterial.Shininess", 76.8f);

	ArrowShader.SetVec3("USpotLight.Light.Ambient", glm::vec3(1.f));
	ArrowShader.SetVec3("USpotLight.Light.Diffuse", glm::vec3(1.f));
	ArrowShader.SetVec3("USpotLight.Light.Specular", glm::vec3(1.f));
	ArrowShader.Set1f("USpotLight.Constant", 1.f);
	ArrowShader.Set1f("USpotLight.Linear", 0.09f);
	ArrowShader.Set1f("USpotLight.Quadratic", 0.032f);
	ArrowShader.Set1f("USpotLight.CutOff", glm::cos(glm::radians(22.5f)));
	ArrowShader.Set1f("USpotLight.OuterCutOff", glm::cos(glm::radians(25.f)));
}

void ImGuiUpdate()
{
	ImGui::CaptureMouseFromApp(false);
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void ProcessInput(GLFWwindow *Window)
{
	if (CurrentState == EState::OnGame)
	{
		if (glfwGetKey(Window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		{
			DeltaTime *= 3.f;
		}
		if (glfwGetKey(Window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		{
			DeltaTime /= 3.f;
		}
		if (glfwGetKey(Window, GLFW_KEY_W) == GLFW_PRESS)
		{
			Camera.ProcessKeyboard(ECameraMovement::Forward, DeltaTime);
		}
		if (glfwGetKey(Window, GLFW_KEY_S) == GLFW_PRESS)
		{
			Camera.ProcessKeyboard(ECameraMovement::Backward, DeltaTime);
		}
		if (glfwGetKey(Window, GLFW_KEY_A) == GLFW_PRESS)
		{
			Camera.ProcessKeyboard(ECameraMovement::Left, DeltaTime);
		}
		if (glfwGetKey(Window, GLFW_KEY_D) == GLFW_PRESS)
		{
			Camera.ProcessKeyboard(ECameraMovement::Right, DeltaTime);
		}
		if (glfwGetKey(Window, GLFW_KEY_E) == GLFW_PRESS)
		{
			Camera.ProcessKeyboard(ECameraMovement::Up, DeltaTime);
		}
		if (glfwGetKey(Window, GLFW_KEY_Q) == GLFW_PRESS)
		{
			Camera.ProcessKeyboard(ECameraMovement::Down, DeltaTime);
		}
	}
}

// Callbacks

void FramebufferSizeCallback(GLFWwindow* Window, int Width, int Height)
{
	glViewport(0, 0, Width, Height);
}

void KeyCallback(GLFWwindow* Window, int Key, int ScanCode, int Action, int Mods)
{
	if (CurrentState == EState::OnGame)
	{
		if (Key == GLFW_KEY_ESCAPE && Action == GLFW_PRESS)
		{
			//glfwSetWindowShouldClose(Window, true);
			glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			CurrentState = EState::OnMenu;
		}
	}
	else if (CurrentState == EState::OnMenu)
	{
		if (Key == GLFW_KEY_ESCAPE && Action == GLFW_PRESS)
		{
			glfwSetInputMode(Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			CurrentState = EState::OnGame;
			FirstMouse = true;
		}
		ImGui_ImplGlfw_KeyCallback(Window, Key, ScanCode, Action, Mods);
	}
	else if (CurrentState == EState::OnDemo)
	{
		if (Key == GLFW_KEY_ESCAPE && Action == GLFW_PRESS)
		{
			CurrentState = EState::OnMenu;
			bDLDemo = false;
			bPLDemo = false;
			bSLDemo = false;
		}
	}

	if (Key == GLFW_KEY_H && Action == GLFW_PRESS)
	{
		bShowHelp = !bShowHelp;
	}

	if (Key == GLFW_KEY_Y && Action == GLFW_PRESS && bSphereTest)
	{
		bWireframe = !bWireframe;
	}

	if (Key == GLFW_KEY_R && Action == GLFW_PRESS && !bSphereTest)
	{
		bReset = true;
	}

	if (Key == GLFW_KEY_T && Action == GLFW_PRESS)
	{
		bSphereTest = !bSphereTest;
	}

	if (Key == GLFW_KEY_F && Action == GLFW_PRESS)
	{
		bFixDirectionalLight = !bFixDirectionalLight;
	}

	if (Key == GLFW_KEY_C && Action == GLFW_PRESS)
	{
		bUseCPU = !bUseCPU;
	}

	if (Key == GLFW_KEY_SPACE && Action == GLFW_PRESS)
	{
		bPause = !bPause;
	}
}

void CharacterCallback(GLFWwindow* Window, unsigned int Codepoint)
{
	if (CurrentState == EState::OnMenu)
	{
		ImGui_ImplGlfw_CharCallback(Window, Codepoint);
	}
}

void MouseCallback(GLFWwindow* Window, double LocationX, double LocationY)
{
	if (CurrentState == EState::OnGame)
	{
		if (FirstMouse)
		{
			LastX = (float)LocationX;
			LastY = (float)LocationY;
			FirstMouse = false;
		}

		float OffsetX = (float)LocationX - LastX;
		float OffsetY = LastY - (float)LocationY; // reversed since y-coordinates go from bottom to top

		LastX = (float)LocationX;
		LastY = (float)LocationY;

		Camera.ProcessMouseMovement(OffsetX, OffsetY);
	}
}

void ScrollCallback(GLFWwindow* Window, double OffsetX, double OffsetY)
{
	if (CurrentState == EState::OnGame)
	{
		Camera.ProcessMouseScroll((float)OffsetY);
	}
	else if (CurrentState == EState::OnMenu)
	{
		ImGui_ImplGlfw_ScrollCallback(Window, OffsetX, OffsetY);
	}
}

void ErrorCallback(int Error, const char* Description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", Error, Description);
}

// ImGui

bool SliderRotation(const char* label, void* v)
{
	ImGuiWindow* Window = ImGui::GetCurrentWindow();

	ImGuiContext& g = *GImGui;
	bool value_changed = false;
	ImGui::BeginGroup();
	ImGui::PushID(label);
	ImGui::PushMultiItemsWidths(2);
	size_t type_size = sizeof(float);
	const char* Formats[3] = { "Yaw:%.3f" , "Pitch:%.3f" };
	float v_min = 0.f;
	float v_max = 359.998993f;
	for (int i = 0; i < 2; i++)
	{
		ImGui::PushID(i);
		value_changed |= ImGui::SliderScalar("##v", ImGuiDataType_Float, v, &v_min, &v_max, Formats[i], 1.f);
		ImGui::SameLine(0, g.Style.ItemInnerSpacing.x);
		ImGui::PopID();
		ImGui::PopItemWidth();
		v = (void*)((char*)v + type_size);
	}
	ImGui::PopID();

	ImGui::TextUnformatted(label, ImGui::FindRenderedTextEnd(label));
	ImGui::EndGroup();
	return value_changed;
}

void ShowHelp(int ScreenWidth)
{
	std::ostringstream X;
	X.precision(3);
	X << std::fixed;
	X << "X: " << Camera.Location.x;

	std::ostringstream Y;
	Y.precision(3);
	Y << std::fixed;
	Y << "Y: " << Camera.Location.y;

	std::ostringstream Z;
	Z.precision(3);
	Z << std::fixed;
	Z << "Z: " << Camera.Location.z;

	std::ostringstream Yaw;
	Yaw.precision(3);
	Yaw << std::fixed;
	Yaw << "Yaw: " << Camera.Yaw;

	std::ostringstream Pitch;
	Pitch.precision(3);
	Pitch << std::fixed;
	Pitch << "Pitch: " << Camera.Pitch;

	std::ostringstream FOV;
	FOV << "FOV: " << Camera.Zoom;

	std::ostringstream FPS;
	FPS << "FPS: " << ImGui::GetIO().Framerate;

	ImGuiStyle& Style = ImGui::GetStyle();
	ImGuiContext* Context = ImGui::GetCurrentContext();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.f, 0.f));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));

	ImGuiWindowFlags WindowFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
	ImGui::SetNextWindowPos(ImVec2(ScreenWidth - 300.f, 0.f));
	ImGui::SetNextWindowSize(ImVec2(300, 0.f), ImGuiCond_Once);
	ImGui::Begin("Info", NULL, WindowFlags);

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.f, 0.f, 0.f, 1.f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.f, 0.f, 0.f, 1.f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.f, 0.f, 0.f, 1.f));
	ImGui::Button(X.str().c_str(), ImVec2(100.f, 0.f)); ImGui::SameLine();
	ImGui::PopStyleColor(3);

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 1.f, 0.f, 1.f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.f, 1.f, 0.f, 1.f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 1.f, 0.f, 1.f));
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.f, 0.f, 0.f, 1.f));
	ImGui::Button(Y.str().c_str(), ImVec2(100.f, 0.f)); ImGui::SameLine();
	ImGui::PopStyleColor(4);

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 1.f, 1.f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.f, 0.f, 1.f, 1.f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 1.f, 1.f));
	ImGui::Button(Z.str().c_str(), ImVec2(100.f, 0.f));
	ImGui::PopStyleColor(3);

	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Style.Colors[ImGuiCol_Button]);
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, Style.Colors[ImGuiCol_Button]);

	ImGui::Button(Yaw.str().c_str(), ImVec2(150.f, 0.f)); ImGui::SameLine();

	ImGui::Button(Pitch.str().c_str(), ImVec2(150.f, 0.f));

	ImGui::Button(FOV.str().c_str(), ImVec2(300.f, 0.f));

	ImGui::PopStyleColor(2);

	FPSValues[FPSValuesOffset] = ImGui::GetIO().Framerate;
	FPSValuesOffset = (FPSValuesOffset + 1) % IM_ARRAYSIZE(FPSValues);
	ImGui::PlotLines("Lines", FPSValues, IM_ARRAYSIZE(FPSValues), FPSValuesOffset, FPS.str().c_str(), FLT_MAX, FLT_MAX, ImVec2(300.f, 2.f * (Context->FontSize + Style.FramePadding.y * 2.f)));

	ImGui::End();

	ImGui::PopStyleVar(4);
}