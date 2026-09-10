# To Do

Make `target_x` and `target_y` reflect in the system prompt

- before refactor/ decouple, `_render_scene()` would move the location both visibly and actually
- now, `render()` in `PyGameDriver` doesn't have access to superclasses so can't move it
- Need to make movement more deterministic

Enforce better thoughts

- `LogitsProcessor` to encourage thoughts of certain sizes
- `LogitsProcessor` to prevent/ discourage thought loops


Tests

- make the LLM say to move, check that the pygame driver moved
- add test that co-ord from LLM appears in next system prompt

Make inter-class communication more lightweight

- toggle between validation and not?
- decorators that're passed a global bool?

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