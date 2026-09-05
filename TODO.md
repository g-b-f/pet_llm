# To Do

## Thought guiding

- remove empty thoughts to prevent loops
- improve thought loop checking algorithm
- remove `idle` and `swim_fast` options
- improve inserted prompt after OOB


## Features
 
- Human can type into tank
- Finger: only if clicked?
- Food, energy
- Make LLM move in cardinal directions instead of to a coordinate
- Simulate vision
    - Raycasting?
- Completely decouple pygame
    - Maybe have a separate class that is inserted to `Tank`, like how `Brain` is inserted to `Tank`?
    - End goal should be no pygame required at all
    - Maybe have option for TUI?