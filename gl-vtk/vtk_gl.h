#include "GL/glew.h"
#include <vtkProp.h>
#include <vtkType.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkObjectFactory.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include "shader.h"
#include <memory>

#define glErrorCheck() \
{ \
    GLenum err = GL_NO_ERROR; \
    while ((err = glGetError()) != GL_NO_ERROR) \
        fprintf(stderr, "GL Error [%s, %d]: %s\n", __FILE__, __LINE__, gluErrorString(err)); \
}

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
    std::shared_ptr<Shader> shader;

private:
    GLPointsProp(const GLPointsProp&) = delete;
    void operator=(const GLPointsProp&) = delete;
};