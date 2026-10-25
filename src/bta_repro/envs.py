"""Small exact environments used to validate learning before simulator runs."""
from dataclasses import dataclass
import torch
from torch import Tensor

# Accepted BPTA source, Tables 1 and 2. Actions retain the paper's A/B/C order.
PAYOFFS = {
    "climbing": [[11., -30., 0.], [-30., 7., 0.], [0., 6., 5.]],
    "penalty": [[-100., 0., 10.], [0., 2., 0.], [10., 0., -100.]],
}

@dataclass(frozen=True)
class MatrixGame:
    """A stateless, one-step, two-player cooperative game.

    One joint action counts as one environment step, regardless of agent count.
    All episodes terminate; there are no time-limit truncations.
    """
    name: str = "climbing"

    def __post_init__(self):
        if self.name not in PAYOFFS:
            raise ValueError(f"Unknown game {self.name!r}; choose {tuple(PAYOFFS)}")

    @property
    def payoff(self) -> Tensor:
        return torch.tensor(PAYOFFS[self.name], dtype=torch.float32)

    @property
    def optimum(self) -> float:
        return self.payoff.max().item()

    def observations(self, batch_size: int, device="cpu") -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.ones(batch_size, 2, 1, device=device), torch.ones(batch_size, 1, device=device)

    def reward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 2 or actions.shape[-1] != 2:
            raise ValueError("actions must have shape [batch, 2]")
        if actions.dtype != torch.long or ((actions < 0) | (actions > 2)).any():
            raise ValueError("actions must contain integer indices 0, 1, or 2")
        return self.payoff.to(actions.device)[actions[:, 0], actions[:, 1]]

    def joint_actions(self, device="cpu") -> Tensor:
        return torch.cartesian_prod(torch.arange(3, device=device), torch.arange(3, device=device))
