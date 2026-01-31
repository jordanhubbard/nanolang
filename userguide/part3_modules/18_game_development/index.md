# Chapter 18: Game Development

**Game loops, physics, and particle systems.**

Essential modules for game development.

## 18.1 Event Handling

```nano
from "modules/event/event.nano" import EventQueue, create_queue, poll, Event

fn process_input() -> bool {
    let queue: EventQueue = (create_queue)
    let event: Event = (poll queue)
    return (== event.type "keydown")
}

shadow process_input {
    assert true
}
```

## 18.2 Vector Math

```nano
from "modules/vector2d/vector2d.nano" import Vec2, add, subtract, length

fn calculate_distance(p1: Vec2, p2: Vec2) -> float {
    let diff: Vec2 = (subtract p2 p1)
    return (length diff)
}

shadow calculate_distance {
    let p1: Vec2 = Vec2 { x: 0.0, y: 0.0 }
    let p2: Vec2 = Vec2 { x: 3.0, y: 4.0 }
    let dist: float = (calculate_distance p1 p2)
    assert (and (> dist 4.9) (< dist 5.1))
}
```

## 18.3 Particle Systems

```nano
from "modules/particles/particles.nano" import create_emitter, update, render

fn particle_demo(window: Window) -> void {
    let emitter: Emitter = (create_emitter 100 100 100)
    (update emitter 0.016)
    (render emitter window)
}

shadow particle_demo {
    assert true
}
```

## Summary

Game development modules:
- ✅ Event handling
- ✅ 2D vector math
- ✅ Particle effects

---

**Previous:** [Chapter 17: OpenGL Graphics](../17_opengl_graphics/index.html)  
**Next:** [Chapter 19: Terminal UI](../19_terminal_ui/index.html)
