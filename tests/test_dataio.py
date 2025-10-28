import numpy as np
import pytest
from rs2simlib.dataio import handle_bullet_file
from rs2simlib.models import DragFunction

from . import data_dir

uscript_dir = data_dir / "UnrealScript"


def test_handle_bullet_file():
    xx_result = handle_bullet_file(
        path=uscript_dir / "TypeXXBullet.uc",
        base_class_name="ROBullet",
    )
    assert xx_result
    assert xx_result.class_name == "TypeXXBullet"
    assert xx_result.parent_name == "ROBullet"
    assert xx_result.speed == 735  # 36750 / 50 == 735. (UU/s -> m/s).
    assert xx_result.damage == 489
    assert (xx_result.damage_falloff == np.array([[302760000, 0.46], [1560250000, 0.12]])).all()
    assert xx_result.drag_func == DragFunction.G7
    assert xx_result.ballistic_coeff == 0.138

    xx_result_partial = handle_bullet_file(
        path=uscript_dir / "TypeXXBullet_Partial.uc",
        base_class_name="ROBullet",  # TODO: what was the point of this again?
    )
    assert xx_result_partial
    assert xx_result_partial.class_name == "TypeXXBullet_Partial"
    assert xx_result_partial.parent_name == "ROBullet_Partial"
    assert xx_result_partial.speed == 0
    assert xx_result_partial.damage == 6969
    assert not xx_result_partial.damage_falloff.any()
    assert xx_result_partial.drag_func == DragFunction.Invalid
    assert xx_result_partial.ballistic_coeff == 0.138

    xx_result_empty = handle_bullet_file(
        path=uscript_dir / "empty.uc",
        base_class_name="ROBullet",
    )
    assert not xx_result_empty

    xx_result_garbage = handle_bullet_file(
        path=uscript_dir / "TypeXXBullet_Garbage.uc",
        base_class_name="ROBullet",
    )
    assert not xx_result_garbage

    with pytest.raises(ValueError):
        _ = handle_bullet_file(
            path=uscript_dir / "TypeXXBullet_WrongName.uc",
            base_class_name="ROBullet",
        )
