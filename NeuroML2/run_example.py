from neuromllite.NetworkGenerator import check_to_generate_or_run
from neuromllite import Simulation

from neuromllite.utils import load_simulation_json

sim = load_simulation_json('SimTestNet.json')

print('Going to run simulation in BindsNet: %s'%sim)

check_to_generate_or_run(['-bindsnet'], sim)
