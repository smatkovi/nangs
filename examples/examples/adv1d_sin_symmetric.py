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

# periodic b.c for the space dimension
boco = PeriodicBoco('boco', {'x': x[:1], 't': t[:-1], 'u': u}, {'x': x[-1:], 't': t[:-1], 'u': u})
pde.addBoco(boco)

# initial condition (dirichlet for temporal dimension)
p00, p0 = np.sin(2.*math.pi*x), np.array([])
for i in u:
    p0 = np.concatenate((p0,p00)) # one for each value of 'u', keeping the order (x, t, u)
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
    hist = pde.solve(epochs=50, path='adv1d_sin_symmetric_best.pth')

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

    # pde.load_state_dict('adv1d_sin_symmetric_best.pth')

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
        _p_ana = np.sin(2. * math.pi * (x - _u * _t))
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
    # ax.set_ylim([90, 110])
    _u = np.average(u)
    pde.evaluate({'x': x, 't': np.array([t[0]]), 'u': np.array([_u])})
    p_nn = pde.outputs['p']
    p_ana = np.sin(2. * math.pi * (x - _u * t[0]))
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








    # evaluate the solution
    # x = np.linspace(0, 1, 50)
    # t = np.linspace(0, 1, 50)
    # u = np.linspace(0, 1, 5)
    # p, p0, l2 = [], [], []
    # for _t in t:
    #     _p, _p0, _l2 = [], [], []
    #     for _u in u:
    #         _p0.append(np.sin(2. * math.pi * (x - _u * _t)))
    #         pde.evaluate({'x': x, 't': np.array([_t]), 'u': np.array([_u])}, device)
    #         _p.append(pde.outputs['p'])
    #         _l2.append(np.mean((pde.outputs['p'] - np.sin(2. * math.pi * (x - _u * _t))) ** 2))
    #     p.append(_p)
    #     p0.append(_p0)
    #     l2.append(_l2)
    #
    # from matplotlib import animation, rc
    #
    # rc('animation', html='html5')
    #
    #
    # def plot(x, p, p0, t, l2, u):
    #     ax.clear()
    #     # tit = ax.set_title(f"t = {t:.2f}, l2 = {l2:.5f}", fontsize=14)
    #     tit = ax.set_title(f"t = {t:.2f}", fontsize=14)
    #     for i, _u in enumerate(u):
    #         print(p0[i])
    #         print(p[i])
    #         ax.plot(x, p0[i], label=f"Exact (u = {_u})")
    #         ax.plot(x, p[i], ".k", label=f"NN (u = {_u}, l2 = {l2[i]:.5f})")
    #     ax.set_xlabel("x", fontsize=14)
    #     ax.set_ylabel("p", fontsize=14, rotation=np.pi / 2)
    #     ax.legend(loc="upper right")
    #     ax.grid(True)
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([-1.2, 1.2])
    #     return [tit]
    #
    #
    # def get_anim(fig, ax, x, p, p0, t, l2, u):
    #     def anim(i):
    #         return plot(x, p[i], p0[i], t[i], l2[i], u)
    #
    #     return anim
    #
    #
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111, autoscale_on=False)
    # animate = get_anim(fig, ax, x, p, p0, t, l2, u=[1])
    # anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=100, blit=False)
    # plt.show()
