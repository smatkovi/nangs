# imports
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
cuda = True
device = "cuda" if torch.cuda.is_available() and cuda else "cpu"

# imports
from nangs.pde import PDE
from nangs.bocos import PeriodicBoco, DirichletBoco
from nangs.solutions import MLP

# define our PDE
class MyPDE(PDE):
    def __init__(self, inputs=None, outputs=None):
        super().__init__(inputs, outputs)
    # the loss of our NN is the PDE !
    def computePDELoss(self, grads, inputs, outputs, params):
        dpdt, dpdx = grads['p']['t'], grads['p']['x']
        u = inputs['u']
        return [dpdt + u*dpdx]

# instanciate pde with keys for inputs/outputs
pde = MyPDE(inputs=['x', 't', 'u'], outputs=['p'])

# define input values for training
x = np.linspace(0,1,30)
t = np.linspace(0,1,30)
u = np.linspace(0,1,20)
pde.setValues({'x': x, 't': t[:-1], 'u': u})

# dirichlet b.c for left boundary
p_l = np.full((len(t) - 1)*len(u), 1.0)
boco = DirichletBoco('dirichlet_left', {'x': x[:1], 't': t[:-1], 'u': u}, {'p': p_l} )
pde.addBoco(boco)
# initial condition (dirichlet for temporal dimension)
p0 = np.zeros(len(x) * len(u))
boco = DirichletBoco('initial_condition', {'x': x, 't': t[:1], 'u': u}, {'p': p0})
pde.addBoco(boco)

# define input values for validation
x_v = np.linspace(0,1,25)
t_v = np.linspace(0,1,15)
u_v = np.linspace(0,1,5)
pde.setValues({'x': x_v, 't': t_v[:-1], 'u': u_v}, train=False)

# define solution topology
mlp = MLP(pde.n_inputs, pde.n_outputs, 5, 2048)
optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-5)
pde.compile(mlp, optimizer)

if __name__ == '__main__':
    # find the solution
    hist = pde.solve(epochs=100, path='adv1d_const_const_best.pth')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(hist['train_loss'], label="train_loss")
    ax1.plot(hist['val_loss'], label="val_loss")
    ax1.grid(True)
    ax1.legend()
    ax1.set_yscale("log")
    for boco in pde.bocos:
        ax2.plot(hist['bocos'][boco.name], label=boco.name)
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale("log")
    plt.show()

    # pde.load_state_dict('adv1d_const_const_best.pth')

    from matplotlib import animation
    from matplotlib.widgets import Slider, Button


    def update(i):
        ax.clear()
        ax.grid(True)
        if i == 0:
            ax.set_xlim([np.min(x), np.max(x)])
        _t = t[i]
        _u = s_u.val
        pde.evaluate({'x': x, 't': np.array([_t]), 'u': np.array([_u])})
        _p = pde.outputs['p']
        ax.plot(x, _p)
        ax.set_title(f"t = {_t:.2f}", fontsize=14)
        return line_nn, tit

    def update_blit(i):
        _t = t[i]
        _u = s_u.val
        pde.evaluate({'x': x, 't': np.array([_t]), 'u': np.array([_u])})
        _p_nn = pde.outputs['p']
        _p_ana = np.where(x < _u * _t, 1, 0)
        line_nn.set_ydata(_p_nn)  # update the data.
        line_ana.set_ydata(_p_ana)  # update the data.
        tit.set_text(f"t = {_t:.2f} s")
        return line_nn, line_ana, tit


    x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    tit = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    ax.set_xlabel("x")
    ax.grid(True)
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([-0.1, 1.1])
    _u = np.average(u)
    pde.evaluate({'x': x, 't': np.array([t[0]]), 'u': np.array([_u])})
    p_nn = pde.outputs['p']
    p_ana = np.where(x < _u * t[0], 1, 0)
    line_nn, = ax.plot(x, p_nn, 'r*')
    line_ana, = ax.plot(x,p_ana)

    anim = animation.FuncAnimation(fig, update_blit, frames=len(t), interval=100, blit=True, repeat=True, repeat_delay=0)
    plt.subplots_adjust(left=0.1, bottom=0.3)

    axcolor = 'lightgoldenrodyellow'
    ax_u = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
    # ax_rho = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
    # ax_cp = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)

    s_u = Slider(ax_u, 'u', np.min(u), np.max(u), valinit=np.average(u), valstep=0.05)

    plt.show()