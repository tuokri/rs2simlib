import pickle
import re
from pathlib import Path
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Union

import numpy as np

from rs2simlib.models import BulletParseResult
from rs2simlib.models import ClassBase
from rs2simlib.models import DragFunction
from rs2simlib.models import PROJECTILE
from rs2simlib.models import WEAPON
from rs2simlib.models import Weapon
from rs2simlib.models import WeaponParseResult

INSTANT_DAMAGE_PATTERN = re.compile(
    r"^\s*InstantHitDamage\(([\w_\d]+)\)\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
PRE_FIRE_PATTERN = re.compile(
    r"^\s*PreFireTraceLength\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
DRAG_FUNC_PATTERN = re.compile(
    r"^\s*DragFunction\s*=\s*([\w\d_]+).*$",
    flags=re.IGNORECASE,
)
BALLISTIC_COEFF_PATTERN = re.compile(
    r"^\s*BallisticCoefficient\s*=\s*([\d.]+).*$",
    flags=re.IGNORECASE,
)
SPEED_PATTERN = re.compile(
    r"^\s*Speed\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
DAMAGE_PATTERN = re.compile(
    r"^\s*Damage\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
FALLOFF_PATTERN = re.compile(
    r"^\s*VelocityDamageFalloffCurve\s*=\s*\(Points\s*=\s*(\(.*\))\).*$",
    flags=re.IGNORECASE,
)
CLASS_PATTERN = re.compile(
    r"^\s*class\s+([\w]+)\s+extends\s+([\w\d_]+)\s*.*[;\n\s]*$",
    flags=re.IGNORECASE,
)
WEAPON_BULLET_PATTERN = re.compile(
    r"^\s*WeaponProjectiles\(([\w_\d]+)\)\s*=\s*class\s*'(.*)'.*$",
    flags=re.IGNORECASE,
)


# def read_weapon_classes(path: Path) -> MutableMapping[str, Weapon]:
#     with path.open("r", encoding="utf-8") as f:
#         data = json.loads(f.read())
#     weapon_classes = {}
#     return weapon_classes

def pdumps_weapon_classes(weapon_classes: MutableMapping[str, Weapon]):
    return pickle.dumps({
        k: v for
        k, v in weapon_classes.items()
        # if k != WEAPON.name
    })


def ploads_weapon_classes(pickle_str: bytes) -> MutableMapping[str, Weapon]:
    return pickle.loads(pickle_str)


def parse_interp_curve(curve: str) -> np.ndarray:
    """The parsed velocity damage falloff curve consists
    of (x,y) pairs, where x is remaining projectile
    speed in m/s and y is the damage scaler at that speed.
    """
    values = []
    curve = re.sub(r"\s", "", curve)
    for match in re.finditer(r"InVal=([\d.]+),OutVal=([\d.]+)", curve):
        # x = math.sqrt(float(match.group(1))) / 50  # UU/s to m/s.
        x = float(match.group(1))
        y = float(match.group(2))
        values.append([x, y])
    return np.array(values)


def strip_comments(text: str):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        flags=re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def check_name(name1: str, name2: str):
    if name1.lower() != name2.lower():
        raise RuntimeError(
            f"class name doesn't match filename: '{name1}' != '{name2}'")


def get_non_comment_lines(path: Path) -> List[str]:
    with path.open("r", encoding="latin-1") as file:
        return [
            d for d in
            strip_comments(file.read()).split("\n")
            if d.strip()
        ]


def handle_weapon_file(path: Path, base_class_name: str
                       ) -> Optional[WeaponParseResult]:
    data = get_non_comment_lines(path)

    if not data:
        return None

    result = WeaponParseResult()
    for line in data:
        if not result.class_name:
            match = CLASS_PATTERN.match(line)
            if match:
                class_name = match.group(1)
                check_name(class_name, path.stem)
                parent_name = match.group(2)
                if not is_weapon_str(parent_name):
                    if class_name == base_class_name:
                        parent_name = base_class_name
                    else:
                        return None
                result.parent_name = parent_name
                result.class_name = class_name
                continue

        if result.pre_fire_length == - 1:
            match = PRE_FIRE_PATTERN.match(line)
            if match:
                result.pre_fire_length = int(match.group(1)) // 50
                continue

        match = WEAPON_BULLET_PATTERN.match(line)
        if match:
            if match.group(1).lower() == "alternate_firemode":
                idx = 1
            else:
                idx = int(match.group(1))
            name = match.group(2)
            result.bullet_names[idx] = name
            continue

        match = INSTANT_DAMAGE_PATTERN.match(line)
        if match:
            if match.group(1).lower() == "alternate_firemode":
                idx = 1
            else:
                idx = int(match.group(1))
            dmg = int(match.group(2))
            result.instant_damages[idx] = dmg
            continue

        # if (result.class_name
        #         and result.bullet_names
        #         and result.parent_name
        #         and result.instant_damages != -1
        #         and result.pre_fire_length != -1):
        #     break
    return result


def handle_bullet_file(path: Path, base_class_name: str
                       ) -> Optional[BulletParseResult]:
    data = get_non_comment_lines(path)

    if not data:
        return None

    result = BulletParseResult()
    for line in data:
        if not result.parent_name:
            match = CLASS_PATTERN.match(line)
            if match:
                class_name = match.group(1)
                check_name(class_name, path.stem)
                parent_name = match.group(2)
                if not is_bullet_str(parent_name):
                    if class_name == base_class_name:
                        parent_name = base_class_name
                    else:
                        return None
                result.class_name = class_name
                result.parent_name = parent_name
                continue
        if result.damage == -1:
            match = DAMAGE_PATTERN.match(line)
            if match:
                result.damage = int(match.group(1))
                continue
        if result.speed == -1:
            match = SPEED_PATTERN.match(line)
            if match:
                result.speed = float(match.group(1)) / 50
                continue
        if not (result.damage_falloff > 0).any():
            match = FALLOFF_PATTERN.match(line)
            if match:
                result.damage_falloff = parse_interp_curve(
                    match.group(1))
                continue
        if result.ballistic_coeff == -1:
            match = BALLISTIC_COEFF_PATTERN.match(line)
            if match:
                result.ballistic_coeff = float(match.group(1))
                continue
        if result.drag_func == DragFunction.Invalid:
            match = DRAG_FUNC_PATTERN.match(line)
            if match:
                result.drag_func = DragFunction(match.group(1))
                continue
        if (result.class_name
                and result.speed != -1
                and result.damage > 0
                and (result.damage_falloff > 0).any()
                and result.ballistic_coeff != -1
                and result.drag_func != DragFunction.Invalid):
            break

    return result


def process_file(path: Path
                 ) -> Optional[Union[BulletParseResult, WeaponParseResult]]:
    path = path.resolve()
    stem = path.stem
    class_str = ""
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            match = CLASS_PATTERN.match(line)
            if match:
                class_str = match.group(0)
    if is_weapon_str(stem) or is_weapon_str(class_str):
        return handle_weapon_file(path, base_class_name=WEAPON.name)
    elif is_bullet_str(stem) or is_bullet_str(class_str):
        return handle_bullet_file(path, base_class_name=PROJECTILE.name)
    else:
        return None


def is_weapon_str(s: str) -> bool:
    s = s.lower()
    return "roweap_" in s or "weapon" in s


def is_bullet_str(s: str) -> bool:
    s = s.lower()
    return "bullet" in s or "projectile" in s


def resolve_parent(
        obj: ClassBase,
        parse_map: MutableMapping,
        class_map: MutableMapping) -> bool:
    parent_name = parse_map[obj.name].parent_name
    obj.parent = class_map.get(parent_name)
    return obj.parent is not None
