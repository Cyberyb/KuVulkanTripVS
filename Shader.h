#pragma once
#include <fstream>
#include <vector>

//Loading a shader
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("=====Failed to open shader file!=====");
    }

    //���仺����
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    //�ص���ͷ��һ���Զ�ȡ�����ֽ�
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}