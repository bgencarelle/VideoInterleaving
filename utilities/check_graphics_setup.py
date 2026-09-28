"""Check GUI graphics bindings without opening a display or creating a window."""
import importlib
import sys


CHECKS = (
    ('glfw', 'the glfw Python binding and system libglfw3 package'),
    ('moderngl', 'moderngl from requirements.txt'),
    ('OpenGL.GL', 'PyOpenGL from requirements.txt'),
)


def main():
    failures = []
    for module, remedy in CHECKS:
        try:
            imported = importlib.import_module(module)
            if module == 'glfw':
                # This calls into libglfw3 but does not initialize a display.
                imported.get_version()
        except Exception as exc:
            failures.append(module)
            print(f'{module}: {exc}\n  Install/check {remedy}', file=sys.stderr)
    if failures:
        return 1
    print('Graphics Python dependencies available (GLFW, ModernGL and PyOpenGL).')
    print('This check does not test an active X11/Wayland display or OpenGL context.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
