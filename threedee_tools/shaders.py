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

        in vec3 v_vert;
        in vec3 v_norm;

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

FARGMENT_SHADER_MULTI_LIGHT = '''
        #version 330

        uniform vec3 RedLightPos;
        uniform vec3 GreenLightPos;
        uniform vec3 Lights;
        uniform vec3 Color;

        in vec3 v_vert;
        in vec3 v_norm;

        out vec4 f_color;

        void main() {
            float lum = clamp(
                dot(
                    normalize(-Lights), 
                    normalize(v_norm)
                ),
                0.0, 
                1.0) * 0.5 + 0.2;
            vec3 combinedLight = Color * lum;
            
            // RED
            lum = clamp(
                dot(
                    normalize(RedLightPos - v_vert), 
                    normalize(v_norm)
                ),
                0.0, 
                1.0) * .5;
            combinedLight += vec3(.9,.25,.25) * lum;
            
            // GREEN
            lum = clamp(
                dot(
                    normalize(GreenLightPos - v_vert), 
                    normalize(v_norm)
                ),
                0.0, 
                1.0) * .5;
            combinedLight += vec3(.3,.9,.3) * lum;
            
            f_color = vec4(combinedLight, 1.0);
        }
    '''
