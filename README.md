![example workflow](https://github.com/wutterfly/polypath/actions/workflows/rust.yml/badge.svg)

# Polypath

"A very basic file parser for .obj and .mtl files."


Allows to read in *.obj* files, extraxt verticies or iterate over contained *objects*, *groups*, *faces* and verticies.



# Missing features:

- smooth shading
  - "s off", "s 1"
- reading .mtl files
- vertex normal calculation


# Supported .obj Features
- verticies ("v )
  + colors
- vertex normals ("vn ")
- vertex texture coords ("vn ")
- objects ("o ")
- groups ("g ")
- faces ("f ")
  - max 4 verticies per face
- comments ("# ")
  - get ignored
- material library ("mtllib ")
- material use ("mtluse ")





# Test Model Sources:
- https://github.com/alecjacobson/common-3d-test-models/tree/master
- "rungholt.obj": https://casual-effects.com/data/