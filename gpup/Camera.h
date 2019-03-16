#pragma once

#include <cmath>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum class ECameraMovement
{
	Forward,
	Backward,
	Left,
	Right,
	Up,
	Down,
};

// Default camera values
const float YAW = 270.0f;
const float PITCH = 0.0f;
const float SPEED = 5.f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class GCamera
{
public:
	// Camera Attributes
	glm::vec3 Location;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;
	// Euler Angles
	float Yaw;
	float Pitch;
	// Camera options
	float MovementSpeed;
	float MouseSensitivity;
	float Zoom;

	// Constructor with vectors
	GCamera(glm::vec3 InLocation = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 InUp = glm::vec3(0.0f, 1.0f, 0.0f), float InYaw = YAW, float InPitch = PITCH);

	// Constructor with scalar values
	GCamera(float LocX, float LocY, float LocZ, float UpX, float UpY, float UpZ, float InYaw, float InPitch);

	// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
	glm::mat4 GetViewMatrix() const;

	// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	void ProcessKeyboard(ECameraMovement Direction, float deltaTime);

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void ProcessMouseMovement(float OffsetX, float OffsetY, GLboolean ConstrainPitch = true);

	// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
	void ProcessMouseScroll(float OffsetY);

	// Calculates the front vector from the Camera's (updated) Euler Angles
	void UpdateCameraVectors();
};

__forceinline GCamera::GCamera(glm::vec3 InLocation, glm::vec3 InUp, float InYaw, float InPitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
	Location = InLocation;
	WorldUp = InUp;
	Yaw = InYaw;
	Pitch = InPitch;
	UpdateCameraVectors();
}

__forceinline GCamera::GCamera(float LocX, float LocY, float LocZ, float UpX, float UpY, float UpZ, float InYaw, float InPitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
	Location = glm::vec3(LocX, LocY, LocZ);
	WorldUp = glm::vec3(UpX, UpY, UpZ);
	Yaw = InYaw;
	Pitch = InPitch;
	UpdateCameraVectors();
}

__forceinline glm::mat4 GCamera::GetViewMatrix() const
{
	return glm::lookAt(Location, Location + Front, Up);
}

__forceinline void GCamera::ProcessKeyboard(ECameraMovement Direction, float DeltaTime)
{
	float Velocity = MovementSpeed * DeltaTime;
	if (Direction == ECameraMovement::Forward)
	{
		Location += Front * Velocity;
	}
	if (Direction == ECameraMovement::Backward)
	{
		Location -= Front * Velocity;
	}
	if (Direction == ECameraMovement::Left)
	{
		Location -= Right * Velocity;
	}
	if (Direction == ECameraMovement::Right)
	{
		Location += Right * Velocity;
	}
	if (Direction == ECameraMovement::Up)
	{
		Location += Up * Velocity;
	}	if (Direction == ECameraMovement::Down)
	{
		Location -= Up * Velocity;
	}
}

__forceinline void GCamera::ProcessMouseMovement(float OffsetX, float OffsetY, GLboolean ConstrainPitch)
{
	OffsetX *= MouseSensitivity;
	OffsetY *= MouseSensitivity;

	Yaw += OffsetX;
	Pitch += OffsetY;

	Yaw = std::fmodf(Yaw + 360.f, 360);
	Pitch = std::fmodf(Pitch + 360.f, 360);

	// Make sure that when pitch is out of bounds, screen doesn't get flipped
	if (ConstrainPitch)
	{
		if (Pitch > 89.f && Pitch < 180.f)
			Pitch = 89.f;
		if (Pitch < 270.f && Pitch > 90.f)
			Pitch = 270.f;
	}

	// Update Front, Right and Up Vectors using the updated Euler angles
	UpdateCameraVectors();
}

__forceinline void GCamera::ProcessMouseScroll(float OffsetY)
{
	if (Zoom >= 1.0f && Zoom <= 179.0f)
		Zoom -= OffsetY;
	if (Zoom <= 1.0f)
		Zoom = 1.0f;
	if (Zoom >= 179.0f)
		Zoom = 179.0f;
}

__forceinline void GCamera::UpdateCameraVectors()
{
	// Calculate the new Front vector
	Front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	Front.y = sin(glm::radians(Pitch));
	Front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	Front = glm::normalize(Front);
	// Also re-calculate the Right and Up vector
	Right = glm::normalize(glm::cross(Front, WorldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
	Up = glm::normalize(glm::cross(Right, Front));
}