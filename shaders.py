import numpy as np
from OpenGL.GL import *

def compile_shaders(vertex_shader_source, fragment_shader_source):
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)

    if glGetShaderiv(vertex_shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise Exception("Vertex shader compilation failed: " + glGetShaderInfoLog(vertex_shader))

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)

    if glGetShaderiv(fragment_shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise Exception("Fragment shader compilation failed: " + glGetShaderInfoLog(fragment_shader))

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        raise Exception("Shader program linking failed: " + glGetProgramInfoLog(shader_program))

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
    vec4 color1 = texture(texture1, TexCoord);
    vec4 color2 = texture(texture2, TexCoord);
    FragColor = color1 * color1.a + color2 * (1.0 - color1.a) * color2.a;
}
"""

shader_program = compile_shaders(vertex_shader_source, fragment_shader_source)

def display_overlay(texture_id1, texture_id2, width, height):
    glUseProgram(shader_program)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id1)
    glUniform1i(glGetUniformLocation(shader_program, "texture1"), 0)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, texture_id2)
    glUniform1i(glGetUniformLocation(shader_program, "texture2"), 1)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(width, 0)
    glTexCoord2f(1, 0)
    glVertex2f(width, height)
    glTexCoord2f(0, 0)
    glVertex2f(0, height)
    glEnd()

    glUseProgram(0)
