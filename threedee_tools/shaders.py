VERTEX_SHADER_NORMAL = '''
        #version 330

        uniform mat4 Mvp;

        in vec3 in_vert;
        in vec3 in_norm;
        //in vec2 in_text;

        out vec3 v_vert;
        out vec3 v_norm;
        //out vec2 v_text;

        void main() {
            gl_Position = Mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
            //v_text = in_text;
        }
    '''

FARGMENT_SHADER_LIGHT_COLOR = '''
        #version 330

        uniform vec3 Lights;
        uniform vec3 Color;
        //uniform bool UseTexture;
        //uniform sampler2D Texture;

        in vec3 v_vert;
        in vec3 v_norm;
        //in vec2 v_text;

        out vec4 f_color;

        void main() {
            float lum = clamp(
                dot(
                    normalize(Lights - v_vert), 
                    normalize(v_norm)
                ),
                0.0, 
                1.0) * 0.6 + 0.4;
            f_color = vec4(Color * lum, 1.0);
        }
    '''
