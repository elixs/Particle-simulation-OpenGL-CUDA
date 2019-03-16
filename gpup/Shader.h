#pragma once

#include <glad/glad.h> // include glad to get all the required OpenGL headers
#include <glm/glm.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "Utils.h"

class GShader
{
public:
	GShader() = default;
	GShader(const char* VertexPath, const char* FragmentPath);
	GShader(int VertexResource, int FragmentResource);

	void InitShader(const char* VertexPath, const char* FragmentPath);
	void Use();
	void SetBool(const char* Name, bool Value) const;
	void Set1i(const char* Name, int Value1) const;
	void Set1f(const char* Name, float Value1) const;
	void Set2f(const char* Name, float Value1, float Value2) const;
	void Set3f(const char* Name, float Value1, float Value2, float Value3) const;
	void Set4f(const char* Name, float Value1, float Value2, float Value3, float Value4) const;
	void Set3fv(const char* Name, float* Vector) const;
	void SetVec3(const char* Name, glm::vec3 Vector) const;
	void SetVec2r(const char* Name, glm::vec2 Rotation) const;
	void SetMatrix4fv(const char* Name, float* Value) const;
	void SetMat4(const char* Name, glm::mat4 Matrix) const;
	void SetLight(std::ostringstream& StringStream, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular);
	void SetDirectionalLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec3 Direction);
	void SetDirectionalLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec2 Direction);
	void SetPointLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec3 Location, float Constant, float Linear, float Quadratic);
	void SetSpotLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec3 Location, glm::vec3 Direction, float Constant, float Linear, float Quadratic, float CutOff, float OuterCutOff);
	void SetProjectionViewModel(glm::mat4 Projection, glm::mat4 View, glm::mat4 Model);

public:
	unsigned int Id;
};

__forceinline GShader::GShader(const char* VertexPath, const char* FragmentPath)
{
	InitShader(FileToChar(VertexPath), FileToChar(FragmentPath));
}

__forceinline GShader::GShader(int VertexResource, int FragmentResource)
{
	HRSRC Vertex = FindResource(NULL, MAKEINTRESOURCE(VertexResource), "SHADER");
	HGLOBAL VertexData = LoadResource(NULL, Vertex);

	HRSRC Fragment = FindResource(NULL, MAKEINTRESOURCE(FragmentResource), "SHADER");
	HGLOBAL FragmentData = LoadResource(NULL, Fragment);

	InitShader((char*)LockResource(VertexData), (char*)LockResource(FragmentData));
}
__forceinline void GShader::InitShader(const char* VertexCode, const char* FragmentCode)
{
	unsigned int Vertex, Fragment;
	int Success;
	char InfoLog[512];

	// Vertex Shader
	Vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(Vertex, 1, &VertexCode, NULL);
	glCompileShader(Vertex);
	glGetShaderiv(Vertex, GL_COMPILE_STATUS, &Success);
	if (!Success)
	{
		glGetShaderInfoLog(Vertex, 512, NULL, InfoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << InfoLog << std::endl;
	};

	// Fragment Shader
	Fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(Fragment, 1, &FragmentCode, NULL);
	glCompileShader(Fragment);
	glGetShaderiv(Fragment, GL_COMPILE_STATUS, &Success);
	if (!Success)
	{
		glGetShaderInfoLog(Fragment, 512, NULL, InfoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << InfoLog << std::endl;
	};

	// Shader Program
	Id = glCreateProgram();
	glAttachShader(Id, Vertex);
	glAttachShader(Id, Fragment);
	glLinkProgram(Id);
	glGetProgramiv(Id, GL_LINK_STATUS, &Success);
	if (!Success)
	{
		glGetProgramInfoLog(Id, 512, NULL, InfoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << InfoLog << std::endl;
	}

	glDeleteShader(Vertex);
	glDeleteShader(Fragment);
}

__forceinline void GShader::Use()
{
	glUseProgram(Id);
}

__forceinline void GShader::SetBool(const char* Name, bool Value) const
{
	glUniform1i(glGetUniformLocation(Id, Name), (int)Value);
}

__forceinline void GShader::Set1i(const char* Name, int Value1) const
{
	glUniform1i(glGetUniformLocation(Id, Name), Value1);
}

__forceinline void GShader::Set1f(const char* Name, float Value1) const
{
	glUniform1f(glGetUniformLocation(Id, Name), Value1);
}

__forceinline void GShader::Set2f(const char* Name, float Value1, float Value2) const
{
	glUniform2f(glGetUniformLocation(Id, Name), Value1, Value2);
}

__forceinline void GShader::Set3f(const char* Name, float Value1, float Value2, float Value3) const
{
	glUniform3f(glGetUniformLocation(Id, Name), Value1, Value2, Value3);
}

__forceinline void GShader::Set4f(const char* Name, float Value1, float Value2, float Value3, float Value4) const
{
	glUniform4f(glGetUniformLocation(Id, Name), Value1, Value2, Value3, Value4);
}

__forceinline void GShader::Set3fv(const char* Name, float* Vector) const
{
	glUniform3fv(glGetUniformLocation(Id, Name), 1, &Vector[0]);
}

__forceinline void GShader::SetVec3(const char* Name, glm::vec3 Vector) const
{
	glUniform3fv(glGetUniformLocation(Id, Name), 1, &Vector[0]);
}

__forceinline void GShader::SetVec2r(const char* Name, glm::vec2 Rotation) const
{
	glm::vec3 Normal = GetNormal(Rotation);
	glUniform3fv(glGetUniformLocation(Id, Name), 1, &Normal[0]);
}

__forceinline void GShader::SetMatrix4fv(const char* Name, float* Value) const
{
	glUniformMatrix4fv(glGetUniformLocation(Id, Name), 1, GL_FALSE, Value);
}

__forceinline void GShader::SetMat4(const char* Name, glm::mat4 Matrix) const
{
	glUniformMatrix4fv(glGetUniformLocation(Id, Name), 1, GL_FALSE, &Matrix[0][0]);
}

__forceinline void GShader::SetLight(std::ostringstream& StringStream, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular)
{
	SetVec3((StringStream.str() + ".Light.Ambient").c_str(), Ambient);
	SetVec3((StringStream.str() + ".Light.Diffuse").c_str(), Diffuse);
	SetVec3((StringStream.str() + ".Light.Specular").c_str(), Specular);
}

__forceinline void GShader::SetDirectionalLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec3 Direction)
{
	std::ostringstream StringStream;

	StringStream << "UDirectionalLight";
	if (Index >= 0)
	{
		StringStream << "s[" << Index << "]";
	}
	SetLight(StringStream, Ambient, Diffuse, Specular);
	SetVec3((StringStream.str() + ".Direction").c_str(), Direction);
}

__forceinline void GShader::SetDirectionalLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec2 Direction)
{
	std::ostringstream StringStream;

	StringStream << "UDirectionalLight";
	if (Index >= 0)
	{
		StringStream << "s[" << Index << "]";
	}
	SetLight(StringStream, Ambient, Diffuse, Specular);
	SetVec2r((StringStream.str() + ".Direction").c_str(), Direction);
}

__forceinline void GShader::SetPointLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec3 Location, float Constant, float Linear, float Quadratic)
{
	std::ostringstream StringStream;

	StringStream << "UPointLight";
	if (Index >= 0)
	{
		StringStream << "s[" << Index << "]";
	}
	SetLight(StringStream, Ambient, Diffuse, Specular);
	SetVec3((StringStream.str() + ".Location").c_str(), Location);
	Set1f((StringStream.str() + ".Constant").c_str(), Constant);
	Set1f((StringStream.str() + ".Linear").c_str(), Linear);
	Set1f((StringStream.str() + ".Quadratic").c_str(), Quadratic);
}

__forceinline void GShader::SetSpotLight(int Index, glm::vec3 Ambient, glm::vec3 Diffuse, glm::vec3 Specular, glm::vec3 Location, glm::vec3 Direction, float Constant, float Linear, float Quadratic, float CutOff, float OuterCutOff)
{
	std::ostringstream StringStream;

	StringStream << "USpotLight";
	if (Index >= 0)
	{
		StringStream << "s[" << Index << "]";
	}
	SetLight(StringStream, Ambient, Diffuse, Specular);
	SetVec3((StringStream.str() + ".Location").c_str(), Location);
	SetVec3((StringStream.str() + ".Direction").c_str(), Direction);
	Set1f((StringStream.str() + ".Constant").c_str(), Constant);
	Set1f((StringStream.str() + ".Linear").c_str(), Linear);
	Set1f((StringStream.str() + ".Quadratic").c_str(), Quadratic);
	Set1f((StringStream.str() + ".CutOff").c_str(), CutOff);
	Set1f((StringStream.str() + ".OuterCutOff").c_str(), OuterCutOff);
}

__forceinline void GShader::SetProjectionViewModel(glm::mat4 Projection, glm::mat4 View, glm::mat4 Model)
{
	SetMat4("UProjection", Projection);
	SetMat4("UView", View);
	SetMat4("UModel", Model);
}
