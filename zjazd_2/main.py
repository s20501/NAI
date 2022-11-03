# Authors: Marcin Żmuda-Trzebiatowski and Jakub Cirocki
# Example: https://github.com/s20501/NAI/blob/main/zjazd_2/example.PNG
#
# The program calculates the count of cubes needed to chill the drink.
# To run program install
# pip install scikit-fuzzy


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership functions
# definitions of temperature, cube size, glass size and cube count
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
cube_size = ctrl.Antecedent(np.arange(1, 4, 1), 'cube_size')
glass_size = ctrl.Antecedent(np.arange(1, 4, 1), 'glass_size')
cube_count = ctrl.Consequent(np.arange(0, 11, 1), 'cube_count')

# Costume-membership functions
cube_count['low'] = fuzz.trimf(cube_count.universe, [0, 0, 2])
cube_count['medium'] = fuzz.trimf(cube_count.universe, [2, 4, 4])
cube_count['high'] = fuzz.trimf(cube_count.universe, [4, 10, 10])

temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 15])
temperature['mild'] = fuzz.trimf(temperature.universe, [15, 40, 40])
temperature['hot'] = fuzz.trimf(temperature.universe, [40, 100, 100])

# Auto-membership functions
glass_size.automf(3, 'quant', ['small', 'medium', 'big'])
cube_size.automf(3, 'quant', ['small', 'medium', 'big'])

# Rules. You need fewer cubes when the cube size is bigger and the glass is smaller
rules = [
    ctrl.Rule(
        temperature['cold'] & (glass_size['small'] | glass_size['medium']) & (cube_size['medium'] | cube_size['big']),
        cube_count['low']),
    ctrl.Rule(
        temperature['cold'] & (glass_size['small'] | glass_size['medium']) & cube_size['small'],
        cube_count['medium']),
    ctrl.Rule(temperature['mild'] & glass_size['medium'] & cube_size['medium'], cube_count['medium']),
    ctrl.Rule(temperature['mild'] & glass_size['big'] & cube_size['small'], cube_count['high']),
    ctrl.Rule(temperature['mild'] & glass_size['small'] & cube_size['big'], cube_count['low']),
    ctrl.Rule(
        temperature['hot'] & (cube_size['small'] | cube_size['medium']) & (glass_size['big'] | glass_size['medium']),
        cube_count['high']),
    ctrl.Rule(
        temperature['hot'] & cube_size['big'] & (glass_size['big'] | glass_size['medium']),
        cube_count['medium']),

    ctrl.Rule(cube_size['small'] & glass_size['small'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['small'] & glass_size['small'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['small'] & glass_size['small'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['small'] & glass_size['medium'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['small'] & glass_size['medium'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['small'] & glass_size['medium'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['small'] & glass_size['big'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['small'] & glass_size['big'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['small'] & glass_size['big'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['medium'] & glass_size['small'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['medium'] & glass_size['small'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['medium'] & glass_size['small'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['medium'] & glass_size['medium'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['medium'] & glass_size['medium'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['medium'] & glass_size['medium'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['medium'] & glass_size['big'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['medium'] & glass_size['big'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['medium'] & glass_size['big'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['big'] & glass_size['small'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['big'] & glass_size['small'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['big'] & glass_size['small'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['big'] & glass_size['medium'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['big'] & glass_size['medium'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['big'] & glass_size['medium'] & temperature['hot'], cube_count['high']),
    ctrl.Rule(cube_size['big'] & glass_size['big'] & temperature['cold'], cube_count['low']),
    ctrl.Rule(cube_size['big'] & glass_size['big'] & temperature['mild'], cube_count['medium']),
    ctrl.Rule(cube_size['big'] & glass_size['big'] & temperature['hot'], cube_count['high']),

]

# Simulating control system
temp_ctrl = ctrl.ControlSystem(rules)
temp = ctrl.ControlSystemSimulation(temp_ctrl)

# Getting inputs
temp.input['temperature'] = int(input("Drink temperature(C): "))
temp.input['cube_size'] = int(input("Ice cube size(1-3): "))
temp.input['glass_size'] = int(input("Glass size(1-3): "))

# Compute
temp.compute()

# Printing outcome
print(temp.output['cube_count'])
