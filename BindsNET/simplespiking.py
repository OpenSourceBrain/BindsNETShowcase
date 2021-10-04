from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from bindsnet.network.monitors import Monitor
from bindsnet.encoding import poisson

import torch
time =10
dt = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_neurons =20

network = Network()
network.add_layer(Input(n=n_neurons), name="X")
network.add_layer(LIFNodes(n=n_neurons), name="Y")
network.add_connection(
    Connection(source=network.layers["X"], target=network.layers["Y"]),
    source="X",
    target="Y",
)

y_monitor = Monitor(
    network.layers["Y"], ["v"], time=int(time / dt), device=device)

network.add_monitor(y_monitor, name="y_monitor")

data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}

print('Starting...')
network.run(inputs=data, time=time)

print('Finished!')



print(y_monitor)
