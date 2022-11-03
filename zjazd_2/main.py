import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(0, 100, 1), 'temperature')
cube_size = ctrl.Antecedent(np.arange(0, 4, 1), 'cube_size')
glass_size = ctrl.Antecedent(np.arange(0, 4, 1), 'glass_size')
cube_count = ctrl.Consequent(np.arange(0, 11, 1), 'cube_count')

cube_count['low'] = fuzz.trimf(cube_count.universe, [0, 0, 2])
cube_count['medium'] = fuzz.trimf(cube_count.universe, [2, 3, 4])
cube_count['high'] = fuzz.trimf(cube_count.universe, [4, 8, 10])

temperature.automf(3, 'quant', ['cold', 'mild', 'hot'])


rule1 = ctrl.Rule(temperature['cold'], cube_count['low'])
rule2 = ctrl.Rule(temperature['mild'], cube_count['medium'])
rule3 = ctrl.Rule(temperature['hot'], cube_count['high'])

temp_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
temp = ctrl.ControlSystemSimulation(temp_ctrl)

temp.input['temperature'] = int(input("Drink temperature(C): "))

temp.compute()

print(temp.output['cube_count'])