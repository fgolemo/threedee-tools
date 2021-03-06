from glumpy import app, gloo, gl

vertex = """
    uniform float theta;
    attribute vec4 color;
    attribute vec2 position;
    varying vec4 v_color;
    void main()
    {
        float ct = cos(theta);
        float st = sin(theta);
        float x = 0.75* (position.x*ct - position.y*st);
        float y = 0.75* (position.x*st + position.y*ct);
        gl_Position = vec4(x, y, 0.0, 1.0);
        v_color = color;
    } """

fragment = """
  varying vec4 v_color;
  void main()
  {
      gl_FragColor = v_color;
  } """

# Build the program and corresponding buffers (with 4 vertices)
quad = gloo.Program(vertex, fragment, count=4)

# Upload data into GPU
quad['color'] = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
quad['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
quad['theta'] = 0

# Create a window with a valid GL context
window = app.Window()

theta = 0.0


# Tell glumpy what needs to be done at each redraw
@window.event
def on_draw(dt):
    window.clear()
    quad['theta'] += dt
    quad.draw(gl.GL_TRIANGLE_STRIP)


@window.event
def on_resize(width,height):
    if width > height:
        x = (width-height)/2
        y = 0
        w = h = height
    else:
        x = 0
        y = (height-width)/2
        w = h = width
    gl.glViewport(int(x), int(y), int(w), int(h))
    window.dispatch_event('on_draw', 0.0)
    window.swap()
    return True

# Run the app
app.run()
