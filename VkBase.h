#pragma once
#include "KuVulkan.h"

/*1.�ڴ���Vulkanʵ��ǰ����AddInstanceLayer(...)��AddInstanceExtension(...)���Ӧ��vector�����ָ������չ���Ƶ�ָ�롣
2.Ȼ������CreateInstance(...)����Vulkanʵ����
3.������Vulkanʵ��ʧ�ܣ���vkCreateInstance(...)����VK_ERROR_LAYER_NOT_PRESENT����InstanceLayers()����һ��instanceLayers��
��CheckInstanceLayers(...)�������ԣ��������õĽ�Ϊ�Ǳ�Ҫ�Ĳ㣬����һ��ȥ���ò���vector����InstanceLayers(...)���Ƹ�instanceLayers��
����VK_ERROR_EXTENSION_NOT_PRESENT����������ơ�Ȼ�����³��Դ���Vulkanʵ����*/

namespace KuVulkan {
	class graphicsBase {
		//������
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
		//���º������ڴ���debug messenger
		VkResult CreateDebugMessenger() {
			/*��Ch1-3���*/
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

		//�ú�������ѡ�������豸ǰ
		void Surface(VkSurfaceKHR surface) {
			if (!this->surface)
				this->surface = surface;
		}


		//���Layer��Extension
		void AddInstanceLayer(const char* layerName) {
			AddLayerOrExtension(instanceLayers, layerName);
		}

		void AddInstanceExtension(const char* extName) {
			AddLayerOrExtension(instanceExtensions, extName);
		}

		//����VkInstance
		VkResult CreateInstance(VkInstanceCreateFlags flags = 0){

		}

		VkResult CheckInstanceExtensions(){

		}


	};
	inline graphicsBase graphicsBase::singleton;
}