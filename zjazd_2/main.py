import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(0, 100, 1), 'temperature')
icecubes = ctrl.Consequent(np.arange(0, 11, 1), 'icecubes')

icecubes['low'] = fuzz.trimf(icecubes.universe, [0, 0, 2])
icecubes['medium'] = fuzz.trimf(icecubes.universe, [0, 2, 4])
icecubes['high'] = fuzz.trimf(icecubes.universe, [4, 8, 10])

temperature.automf(3, 'quant', ['hot', 'mild', 'cold'])

temperature['hot'].view()

rule1 = ctrl.Rule(temperature['cold'], icecubes['low'])
rule2 = ctrl.Rule(temperature['mild'], icecubes['medium'])
rule3 = ctrl.Rule(temperature['hot'], icecubes['high'])

temp_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
temp = ctrl.ControlSystemSimulation(temp_ctrl)

temp.input['temperature'] = 15

temp.compute()

print(temp.output['icecubes'])