# To Do

add hash to StudyReport

Make `target_x` and `target_y` reflect in the system prompt

- before refactor/ decouple, `_render_scene()` would move the location both visibly and actually
- now, `render()` in `PyGameDriver` doesn't have access to superclasses so can't move it
- Need to make movement more deterministic

Enforce better thoughts

- `LogitsProcessor` to encourage thoughts of certain sizes
- `LogitsProcessor` to prevent/ discourage thought loops


Make inter-class communication more lightweight

- toggle between validation and not?
- decorators that're passed a global bool?

## Thought guiding

- remove empty thoughts to prevent loops
- improve thought loop checking algorithm

## Thought logging

Separate logger for sqlite or similar

Include hash


## Features
 
- Human can type into tank
- Finger: only if clicked?
- Food, energy
- Make LLM move in cardinal directions instead of to a coordinate
- Simulate vision
    - Raycasting?
- Add TUI driver
