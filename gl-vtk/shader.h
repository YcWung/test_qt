#include <vtk_glew.h>

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