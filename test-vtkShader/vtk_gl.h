#include "GL/glew.h"
#include <vtkProp.h>
#include <vtkType.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkObjectFactory.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkOpenGLCamera.h>
#include <vtkShaderProgram.h>
#include <vtkShader.h>
#include <vtkOpenGLShaderCache.h>
#include <memory>

#define glErrorCheck() \
{ \
    GLenum err = GL_NO_ERROR; \
    while ((err = glGetError()) != GL_NO_ERROR) \
        fprintf(stderr, "GL Error [%s, %d]: %s\n", __FILE__, __LINE__, gluErrorString(err)); \
}

/**
 * @brief OpenGL context must be initialized before using this class
*/
class GLPointsProp : public vtkProp {
public:
    vtkTypeMacro(GLPointsProp, vtkProp);
    void PrintSelf(std::ostream&, vtkIndent) override;
    static GLPointsProp* New();

    int RenderOpaqueGeometry(vtkViewport*) override;

    void Reserve(const size_t);
    void Upload(const float*, const size_t);

protected:
    GLPointsProp();
    ~GLPointsProp();

    unsigned int vbo, vao;
    size_t capacity, size;

    void InitShaders();
    static const char *vs_code, *fs_code;
    // std::map<vtkShader::Type, vtkShader*> shaders;
    vtkSmartPointer<vtkShaderProgram> program;
    vtkNew<vtkOpenGLShaderCache> shader_cache;

    void SetCameraUniforms(vtkOpenGLCamera*, vtkRenderer*);

private:
    GLPointsProp(const GLPointsProp&) = delete;
    void operator=(const GLPointsProp&) = delete;
};