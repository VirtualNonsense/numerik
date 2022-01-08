from dataclasses import dataclass


@dataclass
class BoundaryCondition:
    location: float


@dataclass
class DirichletBoundaryCondition(BoundaryCondition):
    mu: float


@dataclass
class RobinBoundaryCondition(BoundaryCondition):
    kappa: float
    mu: float
