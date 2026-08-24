from robolab.core.scenes.utils import _usd_quat_wxyz_to_isaaclab_xyzw


def test_usd_identity_quaternion_converts_to_isaaclab3_order():
    assert _usd_quat_wxyz_to_isaaclab_xyzw((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0, 1.0)


def test_usd_quaternion_components_are_reordered_without_sign_change():
    assert _usd_quat_wxyz_to_isaaclab_xyzw((0.1, 0.2, 0.3, 0.4)) == (0.2, 0.3, 0.4, 0.1)
