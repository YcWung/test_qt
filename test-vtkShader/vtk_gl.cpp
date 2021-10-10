#include "vtk_gl.h"

vtkStandardNewMacro(GLPointsProp);

const char* GLPointsProp::vs_code = \
    "#version 330 core\n"
    "in vec4 vertexMC;"
    "uniform mat4 MCDCMatrix;\n"
    "void main() {\n"
    "    gl_Position = MCDCMatrix * vertexMC;"
    "}\n";
const char* GLPointsProp::fs_code = \
    "#version 330 core\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
    "}\n";

GLPointsProp::GLPointsProp() : vbo(0), vao(0), capacity(0), size(0) {
        InitShaders();
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

    shader_cache->ReadyShaderProgram(program);
    int attr_id = program->FindAttributeArray("vertexMC");
    glEnableVertexAttribArray(attr_id);
    glVertexAttribPointer(attr_id, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    this->size = s;
}

int GLPointsProp::RenderOpaqueGeometry(vtkViewport* vp) {
    static bool init = true;
    if (init) {
        init = false;
    }

    auto ren = vtkRenderer::SafeDownCast(vp);
    vtkOpenGLCamera* cam = (vtkOpenGLCamera*)(ren->GetActiveCamera());

    shader_cache->ReadyShaderProgram(program);
    SetCameraUniforms(cam, ren);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(10.0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindVertexArray(vao);

    glDrawArrays(GL_POINTS, 0, size / 3);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return 1;
}

void GLPointsProp::InitShaders() {
    
    vtkNew<vtkShader> v_s;
    v_s->SetSource(vs_code);
    v_s->SetType(vtkShader::Vertex);

    vtkNew<vtkShader> f_s;
    f_s->SetSource(fs_code);
    f_s->SetType(vtkShader::Fragment);

    program = vtkNew<vtkShaderProgram>();
    program->SetVertexShader(v_s);
    program->SetFragmentShader(f_s);
    //shader_cache->ReadyShaderProgram(program);
    //std::cout << "\n\nvs ------------------ :\n\n" << program->GetVertexShader()->GetSource() << "\n";
    //std::cout << "\n\nfs ------------------ :\n\n" << program->GetFragmentShader()->GetSource() << "\n";
}

void GLPointsProp::SetCameraUniforms(vtkOpenGLCamera* cam, vtkRenderer* ren) {
    // [WMVD]C == {world, model, view, display} coordinates
    // E.g., WCDC == world to display coordinate transformation
    vtkMatrix4x4* wcdc;
    vtkMatrix4x4* wcvc;
    vtkMatrix3x3* norms;
    vtkMatrix4x4* vcdc;
    cam->GetKeyMatrices(ren, wcvc, norms, vcdc, wcdc);

    shader_cache->ReadyShaderProgram(program);
    program->SetUniformMatrix("MCDCMatrix", wcdc);
}