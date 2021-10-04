from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.encoding import poisson

network = Network()
network.add_layer(Input(n=n_neurons), name="X")
network.add_layer(LIFNodes(n=n_neurons), name="Y")
network.add_connection(
    Connection(source=network.layers["X"], target=network.layers["Y"]),
    source="X",
    target="Y",
)

data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}
network.run(inputs=data, time=time)

