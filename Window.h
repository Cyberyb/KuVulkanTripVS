#pragma once
#include <GLFW/glfw3.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class VulkanWindow
{
public:
	VulkanWindow();
	VulkanWindow(uint32_t WIDTH, uint32_t HEIGHT);
	~VulkanWindow();

	void InitWindow();
	GLFWwindow* GetWindow();
	void GetWidthAndHeight(uint32_t& widthRef, uint32_t& heightRef);

private:
	uint32_t winWidth;
	uint32_t winHeight;
	GLFWwindow* window;
};

VulkanWindow::VulkanWindow()
{
	winWidth = WIDTH;
	winHeight = HEIGHT;
	InitWindow();
}

VulkanWindow::VulkanWindow(uint32_t WIDTH, uint32_t HEIGHT) :winWidth(WIDTH), winHeight(HEIGHT)
{
	InitWindow();
}

VulkanWindow::~VulkanWindow()
{
}

inline void VulkanWindow::InitWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(winWidth, winHeight, "Vulkan", nullptr, nullptr);
}

inline GLFWwindow* VulkanWindow::GetWindow()
{
	return window;
}

inline void VulkanWindow::GetWidthAndHeight(uint32_t& widthRef, uint32_t& heightRef)
{
	widthRef = winWidth;
	heightRef = winHeight;
}
