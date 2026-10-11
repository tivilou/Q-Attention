"""Named upstream update states; never collapse PTR's two optimizers into one."""
from __future__ import annotations


class NamedStates:
    def __init__(self, **members):
        self.members = members

    def state_dict(self):
        return {name: item.state_dict() for name, item in self.members.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.members):
            raise ValueError("named update states missing/unexpected: both encoder and extra are required")
        for name, item in self.members.items():
            item.load_state_dict(state[name])

    def step(self):
        for item in self.members.values():
            item.step()


class Optimizers(NamedStates):
    def zero_grad(self, **kwargs):
        for item in self.members.values():
            item.zero_grad(**kwargs)


class Schedulers(NamedStates):
    pass
