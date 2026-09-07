# To Do

Make `target_x` and `target_y` reflect in the system prompt

- before refactor/ decouple, `_render_scene()` would move the location both visibly and actually
- now, `render()` in `PyGameDriver` doesn't have access to superclasses so can't move it
- Need to make movement more deterministic

Decouple inference

- eg vLLM
- make it work regardless of file
    - gguf, safetensors, etc

Tests

- make the LLM say to move, check that the pygame driver moved

Lazy load drivers

- using `__lazy_modules__ = ["PyGameDriver"]` etc

Make inter-class communication more lightweight

- toggle between validation and not?

## Thought guiding

- remove empty thoughts to prevent loops
- improve thought loop checking algorithm
- remove `idle` and `swim_fast` options


## Features
 
- Human can type into tank
- Finger: only if clicked?
- Food, energy
- Make LLM move in cardinal directions instead of to a coordinate
- Simulate vision
    - Raycasting?
- Add TUI driver