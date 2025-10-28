import numpy as np
import pytest

from rs2simlib.dataio import handle_bullet_file
from rs2simlib.dataio import strip_comments
from rs2simlib.models import DragFunction
from . import data_dir

uscript_dir = data_dir / "UnrealScript"


def test_handle_bullet_file() -> None:
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


strip_comments_test_data = [
    (
        "",
        "",
    ),
    (
        "/* */",
        " ",
    ),
    (
        "//",
        " ",
    ),
    (
        "" * 500,
        "" * 500,
    ),
    (
        """
        #include <stdio.h> // Inline comment.

        /* Our main function! */
        int main(void)
        {
            // Return some stuff.
            return 0;

            /* This is a comment block
             * // with some nested stuff
             * ?
             * !
             */
        }
        """,
        # NOTE: trailing whitespaces! Important!
        """
        #include <stdio.h>  

         
        int main(void)
        {
             
            return 0;

             
        }
        """,
    ),
    (
        """
        async def main() -> None:
            x = await do_stuff(some_arg)  # Python comment!
            x /= 5
            x = x / 2
            print("/* C comment! */")
        """,
        """
        async def main() -> None:
            x = await do_stuff(some_arg)  # Python comment!
            x /= 5
            x = x / 2
            print("/* C comment! */")
        """,
    ),
]


@pytest.mark.parametrize("src,stripped", strip_comments_test_data)
def test_strip_comments(src: str, stripped: str) -> None:
    result = strip_comments(src)
    assert result == stripped
