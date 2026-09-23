from lib.classes import Brain
from lib.drivers import PyGameDriver
from lib.inference import LlamaCpp
from lib.classes import Tank
from lib.utils import get_logger, values_from_trial
from models import Model, get_model

model_path = get_model(Model.llama)
logger = get_logger(__name__)


if __name__ == "__main__":
    config = values_from_trial(183)
    config.tank.runtime = 60 * 5
    config.brain.params.seed = 11
    config.tank.screen_width, config.tank.screen_height = 1200, 900

    inference = LlamaCpp(config.brain.params, model_path)
    tank_bounds = Tank.get_bounds(config.tank)
    brain = Brain(config.brain, tank_bounds, inference)

    bounds = (config.tank.screen_width, config.tank.screen_height)
    driver = PyGameDriver(config.tank.runtime, bounds)
    # driver = DummyDriver(config.tank.runtime)
    simulation = Tank(brain, config.tank, driver)
    simulation.run()
