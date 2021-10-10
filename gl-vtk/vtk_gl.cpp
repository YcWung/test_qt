#include "vtk_gl.h"

static constexpr const char* vs_fs_code[] = {
    // Vertex Shader Code
    "#version 330 core\n"
    "layout(location = 0) in vec3 position;\n"
    "uniform mat4 MCDCMatrix;\n"
    "void main() {\n"
    "    gl_Position = vec4(position, 1.0);\n"
    "}\n",

    // Fragment Shader Code
    "#version 330 core\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
    "}\n"
};

vtkStandardNewMacro(GLPointsProp);

GLPointsProp::GLPointsProp() : vbo(0), vao(0), capacity(0), size(0) {

}

void GLPointsProp::PrintSelf(std::ostream& o, vtkIndent indent) {
    Superclass::PrintSelf(o, indent);
}

GLPointsProp::~GLPointsProp(){}

void GLPointsProp::Reserve(const size_t capacity) {
    if (vbo != 0) {
        glDeleteBuffers(1, &vbo);
    }
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, capacity * sizeof(float), nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    this->capacity = capacity;
}

void GLPointsProp::Upload(const float* buf, const size_t s) {
    if (vao != 0) {
        glDeleteVertexArrays(1, &vao);
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, s * sizeof(float), buf, GL_STATIC_DRAW);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    this->size = s;
}

int GLPointsProp::RenderOpaqueGeometry(vtkViewport* vp) {
    if (shader == nullptr) {
        shader = std::make_shared<Shader>();
        shader->Compile(vs_fs_code);
    }

    auto ren = vtkRenderer::SafeDownCast(vp);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(10.0);

    shader->Use();

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindVertexArray(vao);

    glDrawArrays(GL_POINTS, 0, size / 3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return 1;
}