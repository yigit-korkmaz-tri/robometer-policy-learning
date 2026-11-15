"""Legacy scene restoration keeps simulator references and visual groups aligned."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py

from robometer_policy_learning.utils.robomimic_scene import restore_dataset_scene


def test_restore_legacy_scene_without_loading_demo_states(tmp_path):
    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.attrs["model_file"] = '''<mujoco><asset>
          <texture name="Cereal_tex-cereal" type="cube"/>
          </asset><worldbody><geom name="floor" type="plane"/>
          <body name="robot0_link0">
          <geom name="robot0_link0_visual" group="1"/>
          <geom name="robot0_link0_collision" group="0" friction="0.95 0.3 0.1"/>
          <site name="robot0_eef_site"/></body></worldbody></mujoco>'''
    robot = SimpleNamespace(naming_prefix="robot0_", models=[])

    class Env:
        model = SimpleNamespace(mujoco_objects=[], mujoco_robots=[robot])
        hard_reset = True

        def __init__(self):
            self.sim_values = [1.0]
            self._obs_cache = {"object_pos": 1.0}
            self._observables = self._setup_observables()

        def _setup_observables(self):
            values = self.sim_values
            return {"object_pos": SimpleNamespace(_sensor=lambda cache: values[0], reset=lambda: None)}

        def modify_observable(self, observable_name, attribute, modifier):
            assert attribute == "sensor"
            self._observables[observable_name]._sensor = modifier

        def edit_model_xml(self, xml):
            return xml

        def reset_from_xml_string(self, xml):
            self.restored = ET.fromstring(xml)
            self.sim_values = [2.0]

    env = Env()
    assert restore_dataset_scene(env, path)
    assert not env.hard_reset
    assert robot._visual_geoms == ["link0_visual"]
    assert robot._contact_geoms == ["link0_collision"]
    assert robot._sites == ["eef_site"]
    assert env.restored.find('.//geom[@name="floor"]').get("group") == "1"
    assert env.restored.find('.//geom[@name="robot0_link0_collision"]').get("friction") == "0.95 0.3 0.1"
    assert env._obs_cache == {}
    assert env._observables["object_pos"]._sensor({}) == 2.0
    env.sim_values[0] = 3.0
    assert env._observables["object_pos"]._sensor({}) == 3.0


def test_missing_scene_leaves_environment_untouched(tmp_path):
    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data/demo_0")
    assert not restore_dataset_scene(object(), path)
