#pragma once
#define GLEW_STATIC
#include "GL/glew.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <string>
#include <cstdio>

void glErrorCheck();
void cudaErrorCheck(cudaError_t err);

struct CuGLTex {
    cudaArray* graphics_array;
    cudaGraphicsResource* graphics_resource; // GL and CUDA shared resource
    GLuint texture_id;       // Main texture ID

    void Create(int width, int height);
    void Delete();
    void Map();
    void Unmap();
    void Bind();
    void Unbind();
};

struct Shader {
    GLuint shaders_program;  // Compiled shaders

    Shader() {}
    Shader(const char* vertex_code, const char* fragment_code) {
        Compile(vertex_code, fragment_code);
    }
    void Compile(const char* vertex_code, const char* fragment_code);
    void Compile(const char* const* vs_fs_code) {
        Compile(vs_fs_code[0], vs_fs_code[1]);
    }
    void Delete();
    void Use();
    void EnableAttrib(const char* var);

};

struct PointsGL {
    GLuint VertexArrayID;
    GLuint vertex_buffer;    // VBO

    void Upload(const GLfloat* points, const size_t pt_num);

    void Delete();
};

struct PointsCuGL {
    cudaGraphicsResource* graphics_resource; // GL and CUDA shared resource
    GLuint vertex_buffer;
    float* cu_ptr;
    size_t buffer_size;

    void Create(size_t point_num);

    void Delete();
};

void SetupCUDA();

bool InitGLEW();