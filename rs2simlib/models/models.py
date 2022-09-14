from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt
from rs2simlib.drag import drag_g1
from rs2simlib.drag import drag_g7

SCALE_FACTOR_INVERSE = 0.065618
SCALE_FACTOR = 15.24

# In UU/s.
MAS_49_GRENADE_RANGE_VELOCITIES = {
    0: 1105,
    1: 1210,
    2: 1310,
    3: 1400,
    4: 1490,
    5: 1570,
    6: 1650,
    7: 1725,
    8: 1795,
    9: 1865,
    10: 1935,
    11: 2000,
    12: 2065,
    13: 2125,
    14: 2190,
    15: 2240,
    16: 2300,
    17: 2360,
    18: 2405,
    19: 2460,
    20: 2515,
    21: 2570,
}


class DragFunction(Enum):
    Invalid = ""
    G1 = "RODF_G1"
    G7 = "RODF_G7"


@dataclass
class ParseResult:
    class_name: str = ""
    parent_name: str = ""

    def __eq__(self, other: "ParseResult"):
        return self.class_name.lower() == other.class_name.lower()


# TODO: NumProjectiles and Spread?
@dataclass
class WeaponParseResult(ParseResult):
    bullet_names: Dict[int, str] = field(default_factory=dict)
    instant_damages: Dict[int, int] = field(default_factory=dict)
    alt_ammo_loadouts: Dict[
        int, "AltAmmoLoadoutParseResult"] = field(default_factory=dict)
    pre_fire_length: int = -1


# TODO: NumProjectiles and Spread?
@dataclass
class AltAmmoLoadoutParseResult(ParseResult):
    # TODO: just build lists directly here?
    bullet_names: Dict[int, str] = field(default_factory=dict)
    instant_damages: Dict[int, int] = field(default_factory=dict)


@dataclass
class BulletParseResult(ParseResult):
    speed: float = -1
    damage: int = -1
    damage_falloff: npt.NDArray[np.float64] = np.array(
        [0.0, 0.0], dtype=np.float64)
    drag_func: DragFunction = DragFunction.Invalid
    ballistic_coeff: float = -1


@dataclass
class ClassBase:
    name: str = field(hash=True)
    parent: Optional["ClassBase"]

    def __eq__(self, other: "ClassBase"):
        return self.name.lower() == other.name.lower()

    def __hash__(self) -> int:
        return hash(self.name)

    # @lru_cache(maxsize=64, typed=True)
    def get_attr(self,
                 attr_name: str,
                 invalid_value: Optional[Any] = None) -> Any:
        if invalid_value is not None:
            attr = getattr(self, attr_name)
            if attr != invalid_value:
                return attr
            parent = self.parent
            while attr == invalid_value:
                attr = getattr(parent, attr_name)
                parent = parent.parent
                if parent.name == parent.parent.name:
                    attr = getattr(parent.parent, attr_name)
                    break
            return attr
        else:
            attr = getattr(self, attr_name)
            if attr:
                return attr
            parent = self.parent
            while not attr:
                attr = getattr(parent, attr_name)
                parent = parent.parent
                if parent.name == parent.parent.name:
                    attr = getattr(parent.parent, attr_name)
                    break
            return attr

    def is_child_of(self, obj: "ClassBase") -> bool:
        if not self.parent:
            return False
        if self.name == obj.name:
            return True
        parent = self.parent
        if parent.name == obj.name:
            return True
        next_parent = parent.parent
        if not next_parent:
            return False
        if next_parent.name == next_parent.parent.name:
            return next_parent.name == obj.name
        return next_parent.is_child_of(obj)

    def find_parent(self, parent_name: str) -> "ClassBase":
        if self.name == parent_name:
            return self
        next_parent = self.parent.parent
        if next_parent.name == next_parent.parent.name:
            if parent_name == next_parent.name:
                return next_parent
            else:
                raise ValueError("parent not found")
        return next_parent.find_parent(parent_name)


@dataclass
class Bullet(ClassBase):
    parent: Optional["Bullet"]
    speed: float
    damage: int
    damage_falloff: npt.NDArray[np.float64]
    drag_func: DragFunction
    ballistic_coeff: float

    def __hash__(self) -> int:
        return super().__hash__()

    def get_speed(self) -> float:
        """Speed (muzzle velocity) in m/s."""
        return self.get_attr("speed", invalid_value=-1)

    def get_speed_uu(self) -> float:
        """Speed in Unreal Units per second (UU/s)."""
        return self.get_speed() * 50

    def get_damage(self) -> int:
        return self.get_attr("damage", invalid_value=-1)

    # @lru_cache(maxsize=64, typed=True)
    def get_damage_falloff(self) -> npt.NDArray[np.float64]:
        dmg_fo = self.damage_falloff
        if (dmg_fo > 0).any():
            return dmg_fo
        parent = self.parent
        while not (dmg_fo > 0).any():
            dmg_fo = parent.damage_falloff
            parent = parent.parent
            if parent.name == parent.parent.name:
                break
        return dmg_fo

    def get_drag_func(self) -> DragFunction:
        return self.get_attr("drag_func", invalid_value=DragFunction.Invalid)

    def get_ballistic_coeff(self) -> float:
        return self.get_attr("ballistic_coeff", invalid_value=-1)


@dataclass
class Weapon(ClassBase):
    parent: Optional["Weapon"]
    bullets: List[Optional[Bullet]]
    instant_damages: List[int]
    pre_fire_length: int
    alt_ammo_loadouts: List[Optional["AltAmmoLoadout"]]

    def __hash__(self) -> int:
        return super().__hash__()

    # @lru_cache(maxsize=64, typed=True)
    def _get_attr_opt_list(self, attr: str) -> Optional[Any]:
        attrs = getattr(self, attr)
        if any(attrs):
            return attrs
        parent = self.parent
        while not any(attrs):
            attrs = getattr(parent, attr)
            parent = parent.parent
            if parent.name == parent.parent.name:
                attrs = getattr(parent.parent, attr)
                break
        return attrs

    def get_bullets(self) -> List[Optional[Bullet]]:
        return self._get_attr_opt_list("bullets")

    def get_bullet(self, index: int) -> Optional[Bullet]:
        return self.get_bullets()[index]

    def get_instant_damages(self) -> List[int]:
        return self._get_attr_opt_list("instant_damages")

    def get_instant_damage(self, index: int) -> Optional[int]:
        return self.get_instant_damages()[index]

    def get_pre_fire_length(self) -> int:
        return self.get_attr("pre_fire_length", invalid_value=-1)

    def get_alt_ammo_loadouts(self) -> List[Optional["AltAmmoLoadout"]]:
        return self._get_attr_opt_list("alt_ammo_loadouts")


@dataclass
class AltAmmoLoadout(ClassBase):
    bullets: List[Optional[Bullet]]
    instant_damages: List[int]

    def __hash__(self) -> int:
        return super().__hash__()


PROJECTILE = Bullet(
    name="Projectile",
    damage=0,
    speed=0,
    damage_falloff=np.array([0.0, 1.0], dtype=np.float64),
    ballistic_coeff=1.0,
    drag_func=DragFunction.G1,
    parent=None,
)
PROJECTILE.parent = PROJECTILE

WEAPON = Weapon(
    name="Weapon",
    bullets=[PROJECTILE],
    parent=None,
    pre_fire_length=50,
    instant_damages=[0],
    alt_ammo_loadouts=[],
)
WEAPON.parent = WEAPON

str_to_df = {
    DragFunction.G1: drag_g1,
    DragFunction.G7: drag_g7,
}


@dataclass
class WeaponSimulation:
    weapon: Weapon
    velocity: np.ndarray = np.array([1, 0], dtype=np.float64)
    location: np.ndarray = np.array([0, 1], dtype=np.float64)
    bullet: Bullet = field(init=False)
    sim: "BulletSimulation" = field(init=False)

    def __post_init__(self):
        self.bullet = self.weapon.get_bullet(0)  # TODO
        self.sim = BulletSimulation(
            bullet=self.bullet,
            velocity=self.velocity.copy(),
            location=self.location.copy(),
        )

    @property
    def distance_traveled_uu(self) -> float:
        return self.sim.distance_traveled_uu

    @property
    def distance_traveled_m(self) -> float:
        return self.distance_traveled_uu / 50

    @property
    def flight_time(self) -> float:
        return self.sim.flight_time

    @property
    def ef_func(self) -> Callable:
        return self.sim.ef_func

    def calc_drag_coeff(self, mach: float) -> float:
        return self.sim.calc_drag_coeff(mach)

    def simulate(self, delta_time: float):
        self.sim.simulate(delta_time)
        self.velocity = self.sim.velocity.copy()
        self.location = self.sim.location.copy()

    def calc_damage(self) -> float:
        return self.sim.calc_damage()


@dataclass
class BulletSimulation:
    bullet: Bullet
    flight_time: float = 0
    bc_inverse: float = 0
    distance_traveled_uu: float = 0
    velocity: np.ndarray = np.array([1, 0], dtype=np.float64)
    location: np.ndarray = np.array([0, 1], dtype=np.float64)
    fo_x: np.ndarray = field(init=False)
    fo_y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.bc_inverse = 1 / self.bullet.get_ballistic_coeff()
        # Initial velocity unit vector * muzzle speed.
        v_normalized = self.velocity / np.linalg.norm(self.velocity)
        if np.isnan(v_normalized).any():
            raise RuntimeError("nan encountered in velocity")
        if np.isinf(v_normalized).any():
            raise RuntimeError("inf encountered in velocity")
        if (v_normalized == 0).all():
            raise RuntimeError("velocity must be non-zero")
        bullet_speed = self.bullet.get_speed_uu()
        # print("bullet_speed =", bullet_speed)
        if np.isnan(bullet_speed):
            raise RuntimeError("nan bullet speed")
        if np.isinf(bullet_speed):
            raise RuntimeError("inf bullet speed")
        if bullet_speed <= 0:
            raise RuntimeError("bullet speed <= 0")
        self.velocity = v_normalized * bullet_speed
        # print("velocity =", self.velocity)
        self.fo_x, self.fo_y = interp_dmg_falloff(self.bullet.get_damage_falloff())

    def ef_func(self, speed_squared_uu: float) -> float:
        return np.interp(
            x=speed_squared_uu,
            xp=self.fo_x,
            fp=self.fo_y,
            left=self.fo_y[0],
            right=self.fo_y[-1])

    def calc_drag_coeff(self, mach: float) -> float:
        return str_to_df[self.bullet.get_drag_func()](mach)

    def simulate(self, delta_time: float):
        if delta_time < 0:
            raise RuntimeError("simulation delta time must be >= 0")
        self.flight_time += delta_time
        if (self.velocity == 0).all():
            return
        v_size = np.linalg.norm(self.velocity)
        v = v_size * SCALE_FACTOR_INVERSE
        mach = v * 0.0008958245617
        cd = self.calc_drag_coeff(mach)
        self.velocity += (
                0.00020874137882624
                * (cd * self.bc_inverse) * np.square(v)
                * SCALE_FACTOR * (-1 * ((self.velocity / v_size) * delta_time)))
        # print("*")
        # print(cd)
        # print(self.bc_inverse)
        # print(v)
        # print(np.square(v))
        # print(self.velocity)
        # print(v_size)
        self.velocity[1] -= (490.3325 * delta_time)
        loc_change = self.velocity * delta_time
        prev_loc = self.location.copy()
        self.location += loc_change
        self.distance_traveled_uu += abs(np.linalg.norm(prev_loc - self.location))

    def calc_damage(self) -> float:
        v_size_sq = np.linalg.norm(self.velocity) ** 2
        power_left = v_size_sq / (self.bullet.get_speed_uu() ** 2)
        damage = self.bullet.get_damage() * power_left
        energy_transfer = self.ef_func(v_size_sq)
        # print("power_left      =", power_left)
        # print("energy_transfer =", energy_transfer)
        damage *= energy_transfer
        return damage


# TODO: Rename to e.g. split_fo.
def interp_dmg_falloff(
        arr: npt.NDArray,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return damage falloff array as two separate
    arrays (x, y).
    """
    harr = np.hsplit(arr, 2)
    x = harr[0].ravel()
    y = harr[1].ravel()
    return x, y
