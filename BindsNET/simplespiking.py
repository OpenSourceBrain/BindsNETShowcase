from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from bindsnet.network.monitors import Monitor
from bindsnet.encoding import poisson
from bindsnet import encoding
import matplotlib.pyplot as plt

from neuromllite.utils import _val_info

import torch
time =1500
dt = 1.0
n_neurons =100

network = Network(dt=dt)

X = Input(n=n_neurons)
Y = LIFNodes(n=n_neurons)
network.add_layer(X, name="X")
network.add_layer(Y, name="Y")
network.add_connection(
    Connection(source=X, target=Y, w=torch.rand(X.n, Y.n)),
    source="X",
    target="Y"
)

# Spike monitor objects.
M1 = Monitor(obj=X, state_vars=['s'])
M2 = Monitor(obj=Y, state_vars=['s'])
network.add_monitor(monitor=M1, name='X')
network.add_monitor(monitor=M2, name='Y')

V2 = Monitor(Y, ['v'])
network.add_monitor(monitor=V2, name='YV')

data = 15 * torch.rand(n_neurons)  # Generate random Poisson rates for 100 input neurons.
train = encoding.poisson(datum=data, time=time)  # Encode input as 5000ms Poisson spike trains.

# Simulate network on generated spike trains.
inputs = {'X' : train}  # Create inputs mapping.

print('Starting...')
network.run(inputs=inputs, time=time)

print('Finished!')


# Plot spikes of input and output layers.
spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

voltages = V2.get("v")
print(_val_info(voltages))
print(voltages.shape)

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for i, layer in enumerate(spikes):

    s = spikes[layer].numpy()
    s = s[0:,0,0:].T

    axes[i].matshow(s, cmap='binary')
    axes[i].set_title('%s spikes' % layer)
    axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
    axes[i].set_xticks(()); axes[i].set_yticks(())
    axes[i].set_aspect('auto')

plt.figure()
for i in range(5):
    v = voltages[0:,0,i]
    plt.plot(v)

plt.tight_layout(); plt.show()
