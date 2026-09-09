import pytest
from lib.tank import Tank
from tests.mocks import BlockingBrain
from lib.drivers import PyGameDriver, DummyDriver
from lib.types.config import TankConfig

def test_drivers_give_same_results():
    config = TankConfig.model_construct()

    py_driver = PyGameDriver(10, (500,500))
    py_brain = BlockingBrain.generate_valid_thoughts(15)
    py_brain.memory.total_recall = True
    py_tank = Tank(py_brain, config, py_driver)

    dummy_driver = DummyDriver(5)
    dummy_brain = BlockingBrain.generate_valid_thoughts(15)
    dummy_brain.memory.total_recall = True
    dummy_tank = Tank(dummy_brain, config, dummy_driver)

    py_tank.run()
    dummy_tank.run()

    py_thoughts = list(py_tank.brain.memory.total_memory)
    dummy_thoughts = list(dummy_tank.brain.memory.total_memory)

    # dummy driver doesn't have "Start exploring"?
    assert py_thoughts == dummy_thoughts
