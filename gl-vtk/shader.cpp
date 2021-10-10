#include "shader.h"
#include <string>
#include <cstdio>

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