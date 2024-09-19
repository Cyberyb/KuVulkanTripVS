#pragma once
#include "KuVulkan.h"

/*1.在创建Vulkan实例前，用AddInstanceLayer(...)和AddInstanceExtension(...)向对应的vector中添加指向层和扩展名称的指针。
2.然后尝试用CreateInstance(...)创建Vulkan实例。
3.若创建Vulkan实例失败，若vkCreateInstance(...)返回VK_ERROR_LAYER_NOT_PRESENT，从InstanceLayers()复制一份instanceLayers，
用CheckInstanceLayers(...)检查可用性，若不可用的仅为非必要的层，创建一份去除该层后的vector，用InstanceLayers(...)复制给instanceLayers。
返回VK_ERROR_EXTENSION_NOT_PRESENT的情况亦类似。然后重新尝试创建Vulkan实例。*/

namespace KuVulkan {
	class graphicsBase {
		//单例类
	private:
		static graphicsBase singleton;
		VkInstance instance;
		std::vector<const char*> instanceLayers;
		std::vector<const char*> instanceExtensions;
		VkSurfaceKHR surface;

		static void AddLayerOrExtension(std::vector<const char*> container, const char* name) {
			for (auto& i : container)
				if (!strcmp(name, i))
					return;
			container.push_back(name);
		}

		graphicsBase() = default;
		graphicsBase(graphicsBase&&) = delete;
		~graphicsBase() {

		}

		VkDebugUtilsMessengerEXT debugMessenger;
		//以下函数用于创建debug messenger
		VkResult CreateDebugMessenger() {
			/*待Ch1-3填充*/
		}

	public:
		//Getter
		static graphicsBase& Base()
		{
			return singleton;
		}
		
		VkInstance Instance() const {
			return instance;
		}

		const std::vector<const char*>& InstanceLayers() const {
			return instanceLayers;
		}

		const std::vector<const char*>& InstanceExtensions() const {
			return instanceExtensions;
		}

		VkSurfaceKHR Surface() const {
			return surface;
		}

		//Setter
		void InstanceLayers(const std::vector<const char*>& layerNames) {
			instanceLayers = layerNames;
		}

		void InstanceExtensions(const std::vector<const char*>& ExtNames) {
			instanceExtensions = ExtNames;
		}

		//该函数用于选择物理设备前
		void Surface(VkSurfaceKHR surface) {
			if (!this->surface)
				this->surface = surface;
		}


		//添加Layer和Extension
		void AddInstanceLayer(const char* layerName) {
			AddLayerOrExtension(instanceLayers, layerName);
		}

		void AddInstanceExtension(const char* extName) {
			AddLayerOrExtension(instanceExtensions, extName);
		}

		//创建VkInstance
		VkResult CreateInstance(VkInstanceCreateFlags flags = 0){

		}

		VkResult CheckInstanceExtensions(){

		}


	};
	inline graphicsBase graphicsBase::singleton;
}