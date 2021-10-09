#include "cu_gl.h"

void glErrorCheck()
{
    GLenum err = GL_NO_ERROR;
    while ((err = glGetError()) != GL_NO_ERROR)
        fprintf(stderr, "GL Error: %s\n", gluErrorString(err));
}

void CuGLTex::Create(int width, int height) {
    // Texture Creation
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    // For cuda gl interop it is need a texture with 4, 2, or 1 floating point
    // component
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
        GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    glErrorCheck();

    cudaErrorCheck(cudaGraphicsGLRegisterImage(
        &graphics_resource, texture_id, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore)
    );
}

void CuGLTex::Delete() {
    glDeleteTextures(1, &texture_id);
    cudaErrorCheck(cudaGraphicsUnregisterResource(graphics_resource));
}

void CuGLTex::Map() {
    cudaErrorCheck(cudaGraphicsMapResources(1, &graphics_resource, 0));

    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(
        &graphics_array, graphics_resource, 0, 0)
    );
}

void CuGLTex::Unmap() {
    cudaErrorCheck(cudaGraphicsUnmapResources(1, &graphics_resource, 0));;
}

void CuGLTex::Bind() {
    glBindTexture(GL_TEXTURE_2D, texture_id);
}

void CuGLTex::Unbind() {
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Shader::Compile(const char* vertex_code, const char* fragment_code) {
    auto compileShader = [](GLuint& shader_id,
        const std::string shader_code) -> bool
    {
        char const* source_ptr = shader_code.c_str();
        glShaderSource(shader_id, 1, &source_ptr, NULL);
        glCompileShader(shader_id);

        GLint result = GL_FALSE;
        int log_length;

        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
        glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &log_length);

        if (log_length > 0)
        {
            GLchar* error_msg = new GLchar[log_length + 1];
            glGetShaderInfoLog(shader_id, log_length, NULL, error_msg);
            fprintf(stderr, "Shaders: %s\n", error_msg);
        }

        return result != GL_FALSE;
    };

    GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    auto vertex_shader_status = compileShader(vertex_shader_id,
        vertex_code);

    GLuint frag_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    auto frag_shader_status = compileShader(frag_shader_id,
        fragment_code);

    if (!vertex_shader_status || !frag_shader_status)
        fprintf(stderr, "Shaders: Could not compile shaders!");

    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader_id);
    glAttachShader(program_id, frag_shader_id);
    glLinkProgram(program_id);

    GLint result = GL_FALSE;
    int log_length;

    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &log_length);

    if (log_length > 0)
    {
        GLchar* error_msg = new GLchar[log_length + 1];
        glGetProgramInfoLog(program_id, log_length, NULL, error_msg);
        fprintf(stderr, "Shaders: %s\n", error_msg);
    }

    glDetachShader(program_id, vertex_shader_id);
    glDetachShader(program_id, frag_shader_id);

    glDeleteShader(vertex_shader_id);
    glDeleteShader(frag_shader_id);

    shaders_program = program_id;
}

void Shader::Delete() {
    glDeleteProgram(shaders_program);
}

void Shader::Use() {
    glUseProgram(shaders_program);
}

void Shader::EnableAttrib(const char* var) {
    GLint loc = glGetAttribLocation(shaders_program, static_cast<const GLchar*>(var));
    glEnableVertexAttribArray(loc);
}

void PointsGL::Upload(const GLfloat* points, const size_t pt_num) {
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * pt_num, points, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glErrorCheck();
}

void PointsGL::Delete() {
    glDeleteBuffers(1, &vertex_buffer);
}

void PointsCuGL::Create(size_t point_num) {
    buffer_size = point_num * 3 * sizeof(GLfloat);
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaErrorCheck(cudaGraphicsGLRegisterBuffer(
        &graphics_resource, vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
    cudaErrorCheck(cudaGraphicsResourceSetMapFlags(graphics_resource, cudaGraphicsMapFlagsWriteDiscard));
    cudaErrorCheck(cudaGraphicsMapResources(1, &graphics_resource, 0));
    cudaErrorCheck(cudaGraphicsResourceGetMappedPointer((void**)&cu_ptr, &buffer_size, graphics_resource));
}

void PointsCuGL::Delete() {
    glDeleteBuffers(1, &vertex_buffer);
}

void SetupCUDA() {
    int device_count;
    cudaErrorCheck(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
        fprintf(stderr, "CUDA Error: No cuda device found");
    else
        cudaErrorCheck(cudaSetDevice(0));
}

bool InitGLEW() {
    auto glewReturn = glewInit();
    if (glewReturn != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW: %s\n",
            glewGetErrorString(glewReturn));
        return false;
    }
    else {
        return true;
    }
}