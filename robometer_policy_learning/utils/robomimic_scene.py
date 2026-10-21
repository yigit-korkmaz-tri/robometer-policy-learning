"""Restore the scene recorded with a robomimic dataset across robosuite versions."""

import xml.etree.ElementTree as ET

import h5py
from loguru import logger


def restore_dataset_scene(env, dataset_path):
    """Load the recorded MJCF while preserving fresh randomized episode resets.

    Robot geom names changed between the original offline_study datasets and
    robosuite 1.4. Refresh the model's cached geom/site names before rebuilding
    simulator references. No recorded simulator states are used for evaluation.
    """
    with h5py.File(dataset_path, "r") as f:
        demo = f["data"][next(iter(f["data"]))]
        xml = demo.attrs.get("model_file")
    if xml is None:
        logger.warning("Dataset has no model_file; retaining the native robosuite scene")
        return False
    source = ET.fromstring(xml)
    # Modern rendering hides collision group 0. Legacy arena XML omitted the
    # floor's group, causing the floor to disappear after loading that XML.
    floor = source.find('.//geom[@name="floor"]')
    if floor is not None and "group" not in floor.attrib:
        floor.set("group", "1")
    # The installed cereal PNG is now a rectangular 2D texture. Its legacy cube
    # declaration fails to compile even when that object is outside the scene.
    for texture in source.findall("./asset/texture"):
        if texture.get("name") == "Cereal_tex-cereal" and texture.get("type") == "cube":
            texture.set("type", "2d")

    models = list(env.model.mujoco_objects)
    for robot in env.model.mujoco_robots:
        models += [robot] + robot.models
    geoms, sites = source.findall(".//geom"), source.findall(".//site")
    for model in models:
        prefix = model.naming_prefix
        matching = [g for g in geoms if g.get("name", "").startswith(prefix)]
        model._visual_geoms = [g.get("name")[len(prefix):] for g in matching if g.get("group", "0") == "1"]
        model._contact_geoms = [g.get("name")[len(prefix):] for g in matching if g.get("group", "0") == "0"]
        model._sites = [s.get("name")[len(prefix):] for s in sites if s.get("name", "").startswith(prefix)]
    env.hard_reset = False
    env.reset_from_xml_string(env.edit_model_xml(ET.tostring(source, encoding="unicode")))
    # ToolHang sensors capture simulator arrays when they are constructed. XML
    # loading replaces the simulator, so refresh sensors as robosuite's hard
    # reset does, while retaining observable activation / sampling settings.
    for name, observable in env._setup_observables().items():
        env.modify_observable(observable_name=name, attribute="sensor", modifier=observable._sensor)
    env._obs_cache = {}
    for observable in env._observables.values():
        observable.reset()
    logger.info("Restored dataset scene XML; subsequent episodes randomize initial conditions normally")
    return True
