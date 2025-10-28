import pickle
import re
from pathlib import Path
from typing import Dict
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt

from rs2simlib.models import BulletParseResult
from rs2simlib.models import ClassBase
from rs2simlib.models import ClassLike
from rs2simlib.models import DragFunction
from rs2simlib.models import PROJECTILE
from rs2simlib.models import WEAPON
from rs2simlib.models import WeaponParseResult
from rs2simlib.models.models import AltAmmoLoadoutParseResult

# TODO: use match-case instead of ifs in parsing functions?
INSTANT_DAMAGE_PATTERN = re.compile(
    r"^\s*InstantHitDamage\(([\w_]+)\)\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
PRE_FIRE_PATTERN = re.compile(
    r"^\s*PreFireTraceLength\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
DRAG_FUNC_PATTERN = re.compile(
    r"^\s*DragFunction\s*=\s*([\w_]+).*$",
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
    r"^\s*class\s+(\w+)\s+extends\s+([\w_]+)\s*.*[;\n\s]*$",
    flags=re.IGNORECASE,
)
WEAPON_BULLET_PATTERN = re.compile(
    r"^\s*WeaponProjectiles\(([\w_]+)\)\s*=\s*class\s*'(.*)'.*$",
    flags=re.IGNORECASE,
)
ALT_AMMO_LOADOUT_PATTERN = re.compile(
    r"AltAmmoLoadouts\s*\((\d)\)\s*=\s*{\s*\n*\(([\w\n\s\[\]='\",._\-/]+)\)[\n\s]*}",
    flags=re.IGNORECASE,
)


def pdumps_class_map(class_map: MutableMapping[str, ClassLike]):
    return pickle.dumps({
        k: v for
        k, v in class_map.items()
    })


def ploads_class_map(pickle_bytes: bytes) -> MutableMapping[str, ClassLike]:
    return pickle.loads(pickle_bytes)


def parse_interp_curve(curve: str) -> npt.NDArray[np.float64]:
    """The parsed velocity damage falloff curve consists
    of (x,y) pairs, where x is the remaining projectile
    speed in m/s and y is the damage scaler at that speed.
    """
    values = []
    curve = re.sub(r"\s", "", curve)
    for match in re.finditer(r"InVal=([\d.]+),OutVal=([\d.]+)", curve):
        # x = math.sqrt(float(match.group(1))) / 50  # UU/s to m/s.
        x = float(match.group(1))
        y = float(match.group(2))
        values.append([x, y])
    return np.array(values, dtype=np.float64)


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


def get_non_comment_lines(data: str) -> List[str]:
    return [
        d for d in
        strip_comments(data).split("\n")
        if d.strip()
    ]


def parse_alt_projectile(attribute_dict: dict, index: int) -> str:
    string = attribute_dict.get(f"WeaponProjectiles[{index}]", "class'None'")
    match = re.match(pattern=r"class'(.*)'", string=string)
    return match.group(1) if match else "None"


def parse_alt_ammo_loadouts(data: List[Tuple[str, str]],
                            class_name: str
                            ) -> Dict[int, AltAmmoLoadoutParseResult]:
    result: Dict[int, AltAmmoLoadoutParseResult] = {}

    for index, alt_lo in data:
        idx = int(index)
        attributes = [x.strip() for x in alt_lo.split(",") if x]
        attributes = [re.sub(r"\s", "", x) for x in attributes]
        attr_tuples = [x.split("=") for x in attributes]
        attrib_dict = {
            key: value for key, value in attr_tuples
        }
        # TODO: refactor.
        if "WeaponContentClassIndex" not in attrib_dict:
            bullet_names = {
                0: parse_alt_projectile(attrib_dict, 0),
                1: parse_alt_projectile(attrib_dict, 1),
            }
            instant_damages = {
                0: int(attrib_dict.get("InstantHitDamage[0]", 0)),
                1: int(attrib_dict.get("InstantHitDamage[1]", 0)),
            }
            alt_class_name = f"{class_name}_AltAmmoLoadouts"
            result[idx] = AltAmmoLoadoutParseResult(
                class_name=alt_class_name,
                parent_name=alt_class_name,
                bullet_names=bullet_names,
                instant_damages=instant_damages,
            )

    return result


def handle_weapon_file(
        path: Path,
        base_class_name: str,
) -> Optional[WeaponParseResult]:
    raw = path.read_text(encoding="latin-1", errors="replace")
    data = get_non_comment_lines(raw)

    if not data:
        return None

    has_alt_ammo = False

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

        if "altammoloadouts" in line.lower():
            if "altammoloadouts.empty" not in line.lower():
                has_alt_ammo = True

        # if (result.class_name
        #         and result.bullet_names
        #         and result.parent_name
        #         and result.instant_damages != -1
        #         and result.pre_fire_length != -1):
        #     break

    if has_alt_ammo:
        matches = ALT_AMMO_LOADOUT_PATTERN.findall(strip_comments(raw))
        if matches:
            result.alt_ammo_loadouts = parse_alt_ammo_loadouts(
                data=matches,
                class_name=result.class_name,
            )

    return result


def handle_bullet_file(
        path: Path,
        base_class_name: str,
) -> Optional[BulletParseResult]:
    raw = path.read_text(encoding="latin-1", errors="replace")
    data = get_non_comment_lines(raw)

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


def process_file(
        path: Path,
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
