#pragma once
#include <GLFW/glfw3.h>
#include <fstream>
#include <sstream>
#include <format>

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
	void TitleFPS();

	bool framebufferResized = false;
private:
	uint32_t winWidth;
	uint32_t winHeight;
	GLFWwindow* pWindow;
	GLFWmonitor* pMonitor;
	const char* windowTitle = "KuVulkan";

	//static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	//{
	//	auto app = reinterpret_cast<VulkanWindow*>(glfwGetWindowUserPointer(window));
	//	app->framebufferResized = true;
	//}
	
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
	//glfwTerminate();
}

inline void VulkanWindow::InitWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	pWindow = glfwCreateWindow(winWidth, winHeight, windowTitle, nullptr, nullptr);
	pMonitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* pMode = glfwGetVideoMode(pMonitor);

	//glfwSetWindowUserPointer(window, this);
	//glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

inline GLFWwindow* VulkanWindow::GetWindow()
{
	return pWindow;
}

inline void VulkanWindow::GetWidthAndHeight(uint32_t& widthRef, uint32_t& heightRef)
{
	widthRef = winWidth;
	heightRef = winHeight;
}

inline void VulkanWindow::TitleFPS()
{
	static double time0 = glfwGetTime();
	static double time1;
	static double dt;
	static int dframe = -1;
	static std::stringstream info;
	time1 = glfwGetTime();
	dframe++;
	if ((dt = time1 - time0) >= 1) {
		info.precision(1);
		info << windowTitle << "    " << std::fixed << dframe / dt << " FPS";
		glfwSetWindowTitle(pWindow, info.str().c_str());
		info.str("");//别忘了在设置完窗口标题后清空所用的stringstream
		time0 = time1;
		dframe = 0;
	}
}
