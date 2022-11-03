import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(0, 100, 1), 'temperature')
ice_cubes = ctrl.Consequent(np.arange(0, 11, 1), 'ice-cubes')

ice_cubes['low'] = fuzz.trimf(ice_cubes.universe, [0, 0, 2])
ice_cubes['medium'] = fuzz.trimf(ice_cubes.universe, [0, 2, 4])
ice_cubes['high'] = fuzz.trimf(ice_cubes.universe, [4, 8, 10])

temperature.automf(3, 'quant', ['cold', 'mild', 'hot'])


rule1 = ctrl.Rule(temperature['cold'], ice_cubes['low'])
rule2 = ctrl.Rule(temperature['mild'], ice_cubes['medium'])
rule3 = ctrl.Rule(temperature['hot'], ice_cubes['high'])

temp_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
temp = ctrl.ControlSystemSimulation(temp_ctrl)

temp.input['temperature'] = int(input("Temperature: "))

temp.compute()

print(temp.output['ice-cubes'])