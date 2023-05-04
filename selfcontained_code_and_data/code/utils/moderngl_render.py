import numpy as np
from utils.array import normalize
from moderngl_window.conf import settings
import moderngl_window as mglw
import moderngl as mgl
from pyrr import Matrix44, Vector3
from PIL import Image
import os

DISPLAY_PRESENT = 'DISPLAY' in os.environ


class HandRenderer():
    def __init__(self, width, height, fx, fy, cx, cy, near, far, F, n_v):
        self._ctx = mgl.create_context(standalone=True)
        
        self._prog = self._create_shader_program()
        self._prog['Light'].value = (0, 0, -10)
        self._prog['Color'].value = (0.67, 0.49, 0.29)
        
        # Note: OpenGL matrices are 16-value arrays with base vectors laid out contiguously in memory. The translation components occupy the 13th, 14th, and 15th elements of the 16-element matrix.
        # Ref: https://www.khronos.org/opengl/wiki/General_OpenGL:_Transformations
        # For constructing transformation matrices refer https://drive.google.com/file/d/1BKWX2T0QNSBpRIKmLB9gx0kXFs3jUmnB/view
        self.mat_view = self._create_view_mat()
        self._width = width
        self._height = height
        self._near = near
        self._far = far
        self.mat_projection = self._create_projection_mat(width, height, fx, fy, cx, cy, self._near, self._far)
        self.mvp = (self.mat_projection @ self.mat_view).astype('f4')
        self._prog['mvp'].write(self.mvp.T.copy())  # GLSL needs matrix in column major
        
        self._vbo_model_pos, self._vbo_model_nor, self._vao_model = self._create_model_vbo_vao(F, n_v)
        self._vbo_bg, self._vao_bg = self._create_bg_vbo_vao()
        self._bg_texture = self._create_bg_texture(width, height)

        self._fbo = self._create_framebuffer(width, height)

        self._ctx.enable(mgl.CULL_FACE | mgl.DEPTH_TEST)

    def _create_shader_program(self):
        return self._ctx.program(
            vertex_shader='''
                #version 330
                
                uniform mat4 mvp;
                uniform bool UseTexture;

                in vec3 in_v_pos;
                in vec3 in_v_nor;
                in vec2 in_texcoord_0;

                out vec3 v_pos;
                out vec3 v_nor;
                out vec2 v_tex;

                void main() {
                    if (UseTexture) {
                        gl_Position = vec4(in_v_pos, 1.0);
                    } else {
                        gl_Position = mvp * vec4(in_v_pos, 1.0);
                    }
                    v_pos = in_v_pos;
                    v_nor = in_v_nor;
                    v_tex = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Light;
                uniform vec3 Color;
                uniform bool UseTexture;
                uniform sampler2D Texture;

                in vec3 v_pos;
                in vec3 v_nor;
                in vec2 v_tex;

                out vec4 f_color;

                void main() {
                    if (UseTexture) {
                        float tex_value = texture(Texture, v_tex).r;
                        f_color = vec4(tex_value, tex_value, tex_value, 1.0);
                    } else {
                        float lum = clamp(dot(normalize(Light - v_pos), normalize(v_nor)), 0.0, 1.0) * 0.8 + 0.2;
                        f_color = vec4(Color * lum, 0.8);
                        // f_color = vec4(0.0, 0.0, 1.0, 0.5);
                    }
                }
            '''
        )

    def _create_view_mat(self):
        # E: eye position
        # A: lookat point
        # v: up vector
        E = np.array([0, 0, 0.0])
        A = np.array([0, 0, 1.0])
        v = np.array([0, -1.0, 0])  # as per realsense camera

        n = - normalize(A - E)
        u = normalize(np.cross(v, n))
        v = normalize(np.cross(n, u))
        mat_view = np.array([
            [u[0], u[1], u[2], -u.T @ E],
            [v[0], v[1], v[2], -v.T @ E],
            [n[0], n[1], n[2], -n.T @ E],
            [   0,    0,    0,    1]
        ])
        
        return mat_view

    def _create_projection_mat(self, width, height, fx, fy, cx, cy, near, far):
        # Ref: https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ/57335955
        
        # X = (u - cu) * Z / fx # cu and cx both are same, principal point
        left = -cx * near / fx  # set u = 0 in above formula, left of image plane has u = 0
        top = cy * near / fy    # top of image plane has v = 0
        right = (width - cx) * near/fx    # right of image plane has u = width
        bottom = -(height - cy) * near/fy   # bottom of image plane has v = height
        mat_projection = np.array([
            [2*near/(right-left),                   0, (right+left)/(right-left),                      0],
            [                  0, 2*near/(top-bottom), (top+bottom)/(top-bottom),                      0],
            [                  0,                   0,    -(far+near)/(far-near), -2*far*near/(far-near)],
            [                  0,                   0,                        -1,                      0]
        ])

        # Note: The below works too, but not as accurate as the above one
        # Ref: https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch04.html
        # fovy = np.rad2deg( 2 * np.arctan(0.5*height/fy) )
        # aspect = (width * fy) / (height * fx)
        # mat_projection = Matrix44.perspective_projection(fovy, aspect, near, far)

        return mat_projection

    def _create_model_vbo_vao(self, F, n_v):
        vbo_model_pos = self._ctx.buffer(reserve=n_v * 3 * 4, dynamic=True)
        vbo_model_nor = self._ctx.buffer(reserve=n_v * 3 * 4, dynamic=True)
        ibo_model = self._ctx.buffer(F.astype('i4'))
        vao_model_content = [
            (vbo_model_pos, '3f', 'in_v_pos'),
            (vbo_model_nor, '3f', 'in_v_nor')
        ]
        vao_model = self._ctx.vertex_array(self._prog, vao_model_content, ibo_model)

        return vbo_model_pos, vbo_model_nor, vao_model

    def _create_bg_vbo_vao(self):
        z = 0
        bg_quad_vertices = np.array([
            # x,  y,  z,  u,  v,         
            -1, -1,  z,  0,  1,  # lower left
             1, -1,  z,  1,  1,  # lower right
             1,  1,  z,  1,  0,  # upper right
            -1,  1,  z,  0,  0,  # upper left
        ])
        bg_quad_indices = np.array([0, 1, 3, 1, 2, 3])
        vbo_bg = self._ctx.buffer(bg_quad_vertices.astype('f4'))
        ibo_bg = self._ctx.buffer(bg_quad_indices.astype('i4'))
        vao_bg_content = [
            (vbo_bg, '3f 2f', 'in_v_pos', 'in_texcoord_0')
        ]
        vao_bg = self._ctx.vertex_array(self._prog, vao_bg_content, ibo_bg)

        return vbo_bg, vao_bg

    def _create_bg_texture(self, width, height):
        return self._ctx.texture((width, height), 1, dtype='f1') # 'f1' is np.uint8

    def _create_framebuffer(self, width, height):
        rgba_texture = self._ctx.texture((width, height), 4, dtype='f1')
        depth_texture = self._ctx.depth_texture((width, height))
        fbo = self._ctx.framebuffer(color_attachments=rgba_texture, depth_attachment=depth_texture)

        return fbo

    def write_vbo(self, img_uint8, V, N):
        self._bg_texture.write(img_uint8.tobytes())
        self._vbo_model_pos.write(V.astype('f4').tobytes())
        self._vbo_model_nor.write(N.astype('f4').tobytes())

    def _render_vao_bg(self):
        self._fbo.depth_mask = False
        self._prog['UseTexture'].value = True
        self._bg_texture.use()
        self._vao_bg.render()

    def _render_vao_model(self):
        self._fbo.depth_mask = True
        self._prog['UseTexture'].value = False
        self._vao_model.render()

    def render(self):
        self._fbo.use()
        self._fbo.clear()

        self._render_vao_bg()
        self._render_vao_model()

    def extract_fbo_color(self):
        # return Image.frombytes('RGB', self._fbo.size, self._fbo.read(attachment=0), 'raw', 'RGB', 0, -1)
        color = np.frombuffer(self._fbo.read(attachment=0, dtype='f1'), dtype=np.uint8).reshape((self._height, self._width, 3))
        color = np.flipud(color)   # flip is  probably required because of the way image is stored as a array (y origin is above)
        return color

    def extract_fbo_depth(self):
        depth = np.frombuffer(self._fbo.read(attachment=-1, dtype='f4'), dtype=np.dtype('f4')).reshape((self._height, self._width))
        depth = np.flipud(depth)
        # depth = np.flipud(depth).copy()
        m_bg = depth == 1
        depth = depth * (self._far - self._near) + self._near   # convert from ndcs to view space depth values
        depth[m_bg] = 0
        return depth*1000

class Pointcloud_and_HandRenderer():
    # def __init__(self, width, height, fx, fy, cx, cy, near, far, F, n_v, x_size, x_color, x_opacity, line_width, line_color, line_opacity):
    def __init__(self, width, height, fx, fy, cx, cy, near, far, F, n_v):
        # self._ctx = mgl.create_context(standalone=True, backend='egl')
        self._ctx = mgl.create_context(standalone=True)
        
        self._prog = self._create_shader_program()
        self._prog['Light'].value = (0, 0, -10)
        # self._prog['Color'].value = (0.67, 0.49, 0.29)
        # self._prog['x_color'].value = x_color
        # self._prog['x_opacity'].value = x_opacity
        # self._ctx.point_size = x_size
        # self._prog['line_color'].value = line_color
        # self._prog['line_opacity'].value = line_opacity
        # self._ctx.line_width = line_width

        
        # Note: OpenGL matrices are 16-value arrays with base vectors laid out contiguously in memory. The translation components occupy the 13th, 14th, and 15th elements of the 16-element matrix.
        # Ref: https://www.khronos.org/opengl/wiki/General_OpenGL:_Transformations
        # For constructing transformation matrices refer https://drive.google.com/file/d/1BKWX2T0QNSBpRIKmLB9gx0kXFs3jUmnB/view
        self.mat_view = self._create_view_mat()
        self._width = width
        self._height = height
        self._near = near
        self._far = far
        self.mat_projection = self._create_projection_mat(width, height, fx, fy, cx, cy, self._near, self._far)
        self.mvp = (self.mat_projection @ self.mat_view).astype('f4')
        self._prog['mvp'].write(self.mvp.T.copy())  # GLSL needs matrix in column major
        
        self._vbo_model_pos, self._vbo_model_nor, self._vao_model = self._create_model_vbo_vao(F, n_v)
        self._vbo_x_dense, self._vao_x_dense = self._create_points_vbo_vao()
        self._vbo_x, self._vbo_xn, self._vao_x = self._create_points_with_normals_vbo_vao()
        self._vbo_y, self._vbo_yn, self._vao_y = self._create_points_with_normals_vbo_vao()
        self._vbo_lines, self._vao_lines = self._create_lines_vbo_vao()
        self._vbo_k, self._vao_k = self._create_points_vbo_vao()

        self._fbo = self._create_framebuffer(width, height)

        self._ctx.enable(mgl.CULL_FACE | mgl.DEPTH_TEST | mgl.BLEND)

    def _create_shader_program(self):
        return self._ctx.program(
            vertex_shader='''
                #version 330
                
                uniform mat4 mvp;

                in vec3 in_v_pos;
                in vec3 in_v_nor;

                out vec3 v_pos;
                out vec3 v_nor;

                void main() {
                    
                    gl_Position = mvp * vec4(in_v_pos, 1.0);
                    
                    v_pos = in_v_pos;
                    v_nor = in_v_nor;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Light;
                uniform vec3 color;
                uniform bool plot_points;
                uniform bool plot_points_with_normal;
                uniform bool plot_lines;
                uniform float opacity;

                in vec3 v_pos;
                in vec3 v_nor;
                in vec2 v_tex;

                out vec4 f_color;

                void main() {
                    if (plot_lines || plot_points) {
                        f_color = vec4(color, opacity);
                    } else {
                        float lum = clamp(dot(normalize(Light - v_pos), normalize(v_nor)), 0.0, 1.0) * 0.8 + 0.2;
                        f_color = vec4(color * lum, opacity);
                        // f_color = vec4(0.0, 0.0, 1.0, 0.5);
                    }
                }
            '''
        )

    def _create_view_mat(self):
        # E: eye position
        # A: lookat point
        # v: up vector
        E = np.array([0, 0, 0.0])
        A = np.array([0, 0, 1.0])
        v = np.array([0, -1.0, 0])  # as per realsense camera

        n = - normalize(A - E)
        u = normalize(np.cross(v, n))
        v = normalize(np.cross(n, u))
        mat_view = np.array([
            [u[0], u[1], u[2], -u.T @ E],
            [v[0], v[1], v[2], -v.T @ E],
            [n[0], n[1], n[2], -n.T @ E],
            [   0,    0,    0,    1]
        ])
        
        return mat_view

    def _create_projection_mat(self, width, height, fx, fy, cx, cy, near, far):
        # Ref: https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ/57335955
        
        # X = (u - cu) * Z / fx # cu and cx both are same, principal point
        left = -cx * near / fx  # set u = 0 in above formula, left of image plane has u = 0
        top = cy * near / fy    # top of image plane has v = 0
        right = (width - cx) * near/fx    # right of image plane has u = width
        bottom = -(height - cy) * near/fy   # bottom of image plane has v = height
        mat_projection = np.array([
            [2*near/(right-left),                   0, (right+left)/(right-left),                      0],
            [                  0, 2*near/(top-bottom), (top+bottom)/(top-bottom),                      0],
            [                  0,                   0,    -(far+near)/(far-near), -2*far*near/(far-near)],
            [                  0,                   0,                        -1,                      0]
        ])

        # Note: The below works too, but not as accurate as the above one
        # Ref: https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch04.html
        # fovy = np.rad2deg( 2 * np.arctan(0.5*height/fy) )
        # aspect = (width * fy) / (height * fx)
        # mat_projection = Matrix44.perspective_projection(fovy, aspect, near, far)

        return mat_projection

    def _create_model_vbo_vao(self, F, n_v):
        vbo_model_pos = self._ctx.buffer(reserve=n_v * 3 * 4, dynamic=True)
        vbo_model_nor = self._ctx.buffer(reserve=n_v * 3 * 4, dynamic=True)
        ibo_model = self._ctx.buffer(F.astype('i4'))
        vao_model_content = [
            (vbo_model_pos, '3f', 'in_v_pos'),
            (vbo_model_nor, '3f', 'in_v_nor')
        ]
        vao_model = self._ctx.vertex_array(self._prog, vao_model_content, ibo_model)

        return vbo_model_pos, vbo_model_nor, vao_model

    def _create_points_vbo_vao(self):
        vbo_points_pos = self._ctx.buffer(reserve='8MB', dynamic=True)
        vao_points_content = [
            (vbo_points_pos, '3f', 'in_v_pos'),
        ]
        vao_points = self._ctx.vertex_array(self._prog, vao_points_content)

        return vbo_points_pos, vao_points

    def _create_points_with_normals_vbo_vao(self):
        vbo_points_pos = self._ctx.buffer(reserve='8MB', dynamic=True)
        vbo_points_nor = self._ctx.buffer(reserve='8MB', dynamic=True)
        vao_points_content = [
            (vbo_points_pos, '3f', 'in_v_pos'),
            (vbo_points_nor, '3f', 'in_v_nor')
        ]
        vao_points = self._ctx.vertex_array(self._prog, vao_points_content)

        return vbo_points_pos, vbo_points_nor, vao_points

    def _create_lines_vbo_vao(self):
        vbo_lines_pos = self._ctx.buffer(reserve='8MB', dynamic=True)
        vao_lines_content = [
            (vbo_lines_pos, '3f', 'in_v_pos'),
        ]
        vao_lines = self._ctx.vertex_array(self._prog, vao_lines_content)

        return vbo_lines_pos, vao_lines

    def _create_framebuffer(self, width, height):
        rgba_texture = self._ctx.texture((width, height), 4, dtype='f1')
        depth_texture = self._ctx.depth_texture((width, height))
        fbo = self._ctx.framebuffer(color_attachments=rgba_texture, depth_attachment=depth_texture)

        return fbo

    def write_vbo_model(self, v, n):
        self._vbo_model_pos.clear()
        self._vbo_model_pos.write(v.astype('f4').tobytes())
        self._vbo_model_nor.clear()
        self._vbo_model_nor.write(n.astype('f4').tobytes())

    def write_vbo_x_dense(self, x_dense):
        self._vbo_x_dense.clear()
        self._vbo_x_dense.write(x_dense.astype('f4').tobytes())

    def write_vbo_k(self, k):
        self._vbo_k.clear()
        self._vbo_k.write(k.astype('f4').tobytes())

    def write_vbo_x(self, x, xn):
        self._vbo_x.clear()
        self._vbo_x.write(x.astype('f4').tobytes())
        self._vbo_xn.clear()
        self._vbo_xn.write(xn.astype('f4').tobytes())

    def write_vbo_y(self, y, yn):
        self._vbo_y.clear()
        self._vbo_y.write(y.astype('f4').tobytes())
        self._vbo_yn.clear()
        self._vbo_yn.write(yn.astype('f4').tobytes())

    def write_vbo_lines(self, x, y):
        # interweave x and y
        # Ref: https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
        line_verts_pos = np.empty((len(x) + len(y), 3), dtype=x.dtype)
        line_verts_pos[0::2] = x
        line_verts_pos[1::2] = y
        self._vbo_lines.clear()
        self._vbo_lines.write(line_verts_pos.astype('f4').tobytes())

    def write_vbo(self, v, n, x_dense, x, xn, y, yn):
        self.write_vbo_model(v, n)
        self.write_vbo_x_dense(x_dense)
        self.write_vbo_x(x, xn)
        self.write_vbo_y(y, yn)
        self.write_vbo_lines(x, y)

    def render_model(self):
        self._fbo.use()
        self._fbo.clear(1, 1, 1)

        self._prog['color'].value = (177/255, 189/255, 180/255) # silver
        self._prog['opacity'].value = 1
        self._vao_model.render(mgl.TRIANGLES)

    def render_x_dense(self):
        self._fbo.use()
        self._fbo.clear(1, 1, 1)

        self._prog['plot_points'].value = True
        self._prog['color'].value = (252/255, 102/255, 3/255)   # orange
        self._prog['opacity'].value = 1
        self._ctx.point_size = 1
        self._vao_x_dense.render(mgl.POINTS)
        self._prog['plot_points'].value = False

    def render_k(self):
        self._fbo.use()
        self._fbo.clear(1, 1, 1)

        self._prog['plot_points'].value = True
        self._prog['color'].value = (0/255, 0/255, 255/255)   # blue
        self._prog['opacity'].value = 1
        self._ctx.point_size = 5
        self._vao_k.render(mgl.POINTS)
        self._prog['plot_points'].value = False

    def render_x(self):
        self._fbo.use()
        self._fbo.clear(1, 1, 1)

        # self._prog['plot_points_with_normal'].value = True
        self._prog['color'].value = (252/255, 102/255, 3/255)   # orange
        self._prog['opacity'].value = 1
        self._ctx.point_size = 3
        self._vao_x.render(mgl.POINTS)
        # self._prog['plot_points_with_normal'].value = False


    def render_x_y_lines_mesh(self):
        # renders x, y, lines, mesh
        self._fbo.use()
        self._fbo.clear(1, 1, 1)

        self._prog['plot_points_with_normal'].value = True
        # x
        self._prog['color'].value = (252/255, 102/255, 3/255)   # orange
        self._prog['opacity'].value = 1
        self._ctx.point_size = 3
        self._vao_x.render(mgl.POINTS)
        
        # y
        self._prog['color'].value = (25/255, 45/255, 245/255) # blue
        self._prog['opacity'].value = 1
        self._ctx.point_size = 3
        self._vao_y.render(mgl.POINTS)

        self._prog['plot_points_with_normal'].value = False


        # lines between x and y
        self._prog['plot_lines'].value = True
        self._prog['color'].value = (225/255, 156/255, 103/255) # light orange
        self._prog['opacity'].value = 1
        self._ctx.line_width = 0.5
        self._vao_lines.render(mgl.LINES)
        self._prog['plot_lines'].value = False

        # mesh
        # self._prog['color'].value = (0.67, 0.49, 0.29) # skin
        self._prog['color'].value = (177/255, 189/255, 180/255) # silver
        self._prog['opacity'].value = 1
        self._vao_model.render(mgl.TRIANGLES)

    def render_x_dense_and_mesh(self):
        # renders x, y, lines, mesh
        self._fbo.use()
        self._fbo.clear(1, 1, 1)

        self._prog['plot_points'].value = True
        self._prog['color'].value = (252/255, 102/255, 3/255)   # orange
        self._prog['opacity'].value = 1
        self._ctx.point_size = 1
        self._vao_x_dense.render(mgl.POINTS)
        self._prog['plot_points'].value = False

        # mesh
        # self._prog['color'].value = (0.67, 0.49, 0.29) # skin
        self._prog['color'].value = (177/255, 189/255, 180/255) # silver
        self._prog['opacity'].value = 1
        self._vao_model.render(mgl.TRIANGLES)


    def extract_fbo_color(self):
        # return Image.frombytes('RGB', self._fbo.size, self._fbo.read(attachment=0), 'raw', 'RGB', 0, -1)
        color = np.frombuffer(self._fbo.read(attachment=0, dtype='f1'), dtype=np.uint8).reshape((self._height, self._width, 3))
        color = np.flipud(color)   # flip is  probably required because of the way image is stored as a array (y origin is above)
        return color

    def extract_fbo_depth(self):
        D = np.frombuffer(self._fbo.read(attachment=-1, dtype='f4'), dtype=np.dtype('f4')).reshape((self._height, self._width))
        D = np.flipud(D)
        m_bg = D == 1
        # Ref: https://ogldev.org/www/tutorial46/tutorial46.html and https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer 
        # let S = -(F+N)/(F-N), T = -2FN/(F-N)
        # as per projection transformation: Z_ndc = (Z*S + T)/-Z
        # opengl transforms ndc which is from [-1, 1] to [0,1], so the actual depth buffer D = (Z_ndc + 1)/2
        # now, we want to obtain Z from D
        # Z_ndc = 2*D - 1
        # Z = -T/(Z_ndc + S)
        Z_ndc =  2*D - 1
        S = self.mat_projection[2, 2]; T = self.mat_projection[2, 3]
        Z = -T / (Z_ndc + S)
        Z = -Z  # camera front is positive in Z, (needs a better explanation, but it works)
        Z[m_bg] = 0
        depth = Z * 1000    # we represent depth in mm
        return depth
