#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "Camera.h"

// Show console
#define  _CONSOLE

#ifdef _CONSOLE
#pragma comment( linker, "/SUBSYSTEM:CONSOLE" )
#else _WINDOWS
#pragma comment( linker, "/SUBSYSTEM:WINDOWS" )
#endif

class GShader;

enum class EState
{
	OnGame,
	OnMenu,
	OnDemo
};
EState CurrentState = EState::OnGame;

// Init
void Init(GLFWwindow* &Window, const char* Title);
void SphereInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &PointLightIndicesSize, int Segments, int Rings, float Radius, void(*Generate)(float*&, int*&, int, int, float, int&, int&));
void ArrowInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &ArrowIndicesSize, int Vertices, float Radius, float Legth, void(*Generate)(float*&, int*&, int, float, float, int&, int&, bool));
void PointLightInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &PointLightIndicesSize, int Segments, int Rings, float Radius, void(*Generate)(float*&, int*&, int, int, float, int&, int&));
void CubeInit(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO, int &CubeIndicesSize, float Lenght, void(*Generate)(float*&, int*&, float, int&, int&));

void SphereShaderInit(GShader &SphereShader);
void ArrowShaderInit(GShader &ArrowShader);

void ImGuiUpdate();

void ProcessInput(GLFWwindow *Window);

// Callbacks
void FramebufferSizeCallback(GLFWwindow* Window, int Width, int Height);
void KeyCallback(GLFWwindow* Window, int Key, int ScanCode, int Action, int Mods);
void CharacterCallback(GLFWwindow* Window, unsigned int Codepoint);
void MouseCallback(GLFWwindow* Window, double LocationX, double LocationY);
void ScrollCallback(GLFWwindow* Window, double OffsetX, double OffsetY);
void ErrorCallback(int Error, const char* Description);


// ImGui
bool SliderRotation(const char* label, void* v);
void ShowHelp(int ScreenWidth);

// Settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
float LastX = SCR_WIDTH / 2.f;
float LastY = SCR_HEIGHT / 2.f;
//
GCamera Camera(glm::vec3(0.f, 0.f, 10.f));

bool FirstMouse = true;

float DeltaTime = 0.f;
float LastFrame = 0.f;
float SphereTime = 0.f;

bool bShowHelp = false;
float FPSValues[120] = { 0 };
int FPSValuesOffset = 0;

bool bDLDemo = false;
bool bPLDemo = false;
bool bSLDemo = false;

bool bWireframe = false;
bool bReset = false;

bool bSphereTest = false;

bool bFixDirectionalLight = true;

bool bUseCPU = false;

bool bPause = false;