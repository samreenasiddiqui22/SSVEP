"""
Shows a coloured cube rotating according to an Explore device's current orientation.
Based on vispy's coloured cube example.
"""
import argparse

import numpy as np

from vispy import app, gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate
from vispy.geometry import create_cube

import explorepy
from explorepy.stream_processor import TOPICS

vertex = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

varying vec4 v_color;
void main()
{
    v_color = color;
    gl_Position = projection * view * model * vec4(position,1.0);
}
"""

fragment = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(512, 512), title='Explore device orientation visualisation',
                            keys='interactive')

        self.index = 0

        # Build cube data
        V, I, _ = create_cube()
        vertices = VertexBuffer(V)
        self.indices = IndexBuffer(I)

        # Build program
        self.program = Program(vertex, fragment)
        self.program.bind(vertices)

        # Build view, model, projection & normal
        view = translate((0, 0, -5))
        model = np.eye(4, dtype=np.float32)
        self.program['model'] = model
        self.program['view'] = view
        self.phi, self.theta = 0, 0
        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True)

        self.activate_zoom()
        self.show()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangles', self.indices)

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]),
                                 2.0, 10.0)
        self.program['projection'] = projection

    def on_mapped_orn_received(self, packet):
        self.program['model'] = packet.matrix
        self.update()


def main():
    parser = argparse.ArgumentParser(description="Example code for orientation visualisation. "
                                                 "Will start with calibration if run for the first time ever.")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device")
    parser.add_argument("--calibrate", action="store_true", help="Force calibration")
    args = parser.parse_args()

    print(args)

    # Create an Explore object
    explore = explorepy.Explore()
    explore.connect(device_name=args.name)

    # Try orientation calibration
    try:
        explore.calibrate_orn(do_overwrite=args.calibrate)
    except AssertionError:
        print("Found calibration data.")

    c = Canvas()

    # Subscribe canvas to mapped orientation packets
    if explore.stream_processor:
        explore.stream_processor.subscribe(callback=c.on_mapped_orn_received, topic=TOPICS.mapped_orn)

    app.run()


if __name__ == '__main__':
    main()

