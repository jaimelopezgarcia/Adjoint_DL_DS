from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import partial
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_damped_data(k = 1, gamma = 0.2, t = np.arange(0,7,0.1), x0 = [2.,2.]):
    kgt = -k
    gammagt = -gamma


    A_ground_truth = np.array([[0.,1.],[kgt,gammagt]])

    def system_gt(x,t):
        dxdt = np.dot(A_ground_truth,x)  
        return dxdt

    tdata =t

    x0_ = [2.,2.]

    data = odeint(system_gt, x0_, tdata)


    xf_gt = data[-1]
    
    
    plt.figure(figsize=(12,8))
    plt.plot(data[:,0],data[:,1])
    plt.plot(data[0,0],data[0,1],'*', label = "IC")
    plt.plot(data[-1,0],data[-1,1],'*', label = "Data to fit")
    plt.legend()
    plt.title("Ground truth damped oscillator")

    return data,xf_gt




def _rk4_step(fun, yk, tk, h):
    
    k1 = fun(yk, tk)
    k2 = fun(yk + h/2*k1, tk + h/2)
    k3 = fun(yk + h/2*k2, tk + h/2)
    k4 = fun(yk + h*k3, tk + h)
    
    yk_next = yk + h/6*(k1+2*k2+2*k3+k4)
    
    return yk_next


def rk4(fun, y0, t, retain_grad = False):
        
    y = []
  
    h = t[1]-t[0]
    yk = y0
    y.append(yk)
    
    for i in range(1,len(t)):
        yknext = _rk4_step(fun, yk, t[i-1], h)
        yk = yknext
        
        if retain_grad:
            yk.retain_grad()
            
        y.append(yk)

    return y

def make_system():

    def system_linear(x,t,params):

        k,gamma = params["k"], params["gamma"]

        dx1dt = x[0][1]
        dx2dt = -k*x[0][0]-gamma*x[0][1]

        return torch.cat([dx1dt.view(-1,1), dx2dt.view(-1,1)], dim = 1)



    k = torch.tensor([1]) 
    gamma = torch.nn.Parameter( torch.tensor([0.01]) )



    params = {"k":k,"gamma":gamma}

    system = partial(system_linear, params = params)
    
    return system, params



def forward_pass(x0,T, system):
    
    out = rk4(system, x0, T, retain_grad = True)
    
    return out

def backward_pass(xpred, xdata):
    
    loss = torch.mean( torch.square(xpred-xdata))
    
    loss.backward()
    
    return loss

def make_numpy(out: list):
    
    with torch.no_grad():
    
        out = out.detach().numpy()
    
    return out

def make_grads_numpy(out: list):
    """ picked in inverse order """
    
    with torch.no_grad():
        
        grads = [out[-i-1].grad.numpy() for i in range(len(out))]
        grads = np.concatenate(grads,axis=0)
    
    return grads







def training_plot(data,register):


    # Create figure
    fig = make_subplots(cols = 2, rows = 1)

    outs,grads = register["outs"], register["grads"]

    N = len(outs)


    fig.add_trace(go.Scatter(), row = 1, col = 1)
    fig.add_trace(go.Scatter(), row = 1, col = 2)
    fig.add_trace(go.Scatter(x = data[1:,0], y = data[1:,1]), row = 1, col = 1)


    layout=go.Layout(
        width = 1200,
        height = 800,
        xaxis=dict(range=[-4, 4], autorange=False, zeroline=False),
        yaxis=dict(range=[-4, 4], autorange=False, zeroline=False),
        xaxis2=dict(range=[-4, 4], autorange=False, zeroline=False),
        yaxis2=dict(range=[-4, 4], autorange=False, zeroline=False),
     #   title_text="Training", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])])
    fig.update_layout(layout)


    frames=[go.Frame(
                data=[go.Scatter(
                        x = outs[k][:,0],
                        y = outs[k][:,1],
                        mode="lines"),

                    go.Scatter(
                        x = grads[k][:,0],
                        y = grads[k][:,1],
                        mode="lines")])

            for k in np.arange(0,N,1)[::1]]

    fig.frames = frames
    
    return fig



def make_figure_fwd_bwd(x0,T,data, indices_data):
    
    
    indices_data = indices_data

    xdata =data[indices_data]

    xdata = torch.tensor(xdata)


    system, params = make_system()

    out = forward_pass(x0,T,system)

    xpred = torch.cat([out[i] for i in indices_data], dim = 0)

    loss = backward_pass(xpred,xdata)

    _out = torch.cat(out, dim = 0)

    fwd_pass = make_numpy(_out[1:])

    bwd_pass = make_grads_numpy(out[1:])

    fig = make_subplots( cols = 2, rows = 1, subplot_titles = ("Forward Pass", "Backward Pass"))


    fig.add_trace( go.Scatter( name = "forward pass/state"), row = 1, col = 1)
    fig.add_trace( go.Scatter( name = "backward pass/adjoint state"), row = 1, col = 2)
    fig.add_trace( go.Scatter( x = data[indices_data,0], y = data[indices_data,1], mode = "markers",name = "Data to fit"), row = 1 ,col = 1)
    fig.add_trace( go.Scatter( x = data[:,0], y = data[:,1], mode = "lines", name = "GT trajectory"), row = 1, col = 1)

    axis = dict(range=[-4, 4], autorange=False, zeroline=False)

    layout = go.Layout(
                    width = 1400,
                    height = 800,
                    xaxis = axis,
                    yaxis = axis,
                    xaxis2 = axis,
                    yaxis2 = axis,
                    updatemenus=[dict(type="buttons",
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None])])])

    fig.update_layout(layout)

    N = len(fwd_pass)
    
    frames = []

    for k in np.arange(0,2*N,1):


        x_start, y_start = data[-1,0], data[-1,1]
        x_end, y_end = x_start+bwd_pass[0,0],y_start+bwd_pass[0,1]

        _arrow_dict = dict(
                        x= x_end,
                        y= y_end,
                        xref="x", yref="y",
                        text=r'$-\partial_{x}f = \lambda_{T}$',
                        showarrow=True,
                        axref = "x", ayref="y",
                        ax= x_start,
                        ay= y_start,
                        arrowhead = 3,
                        align="center",
                        arrowcolor='rgb(255,51,0)',)

        arrow1= go.layout.Annotation(dict(
                    _arrow_dict
                    ))

        layout1 = go.Layout(annotations = [arrow1])


        _arrow_dict["xref"] , _arrow_dict["yref"]= "x2", "y2"
        _arrow_dict["axref"] , _arrow_dict["ayref"]= "x2", "y2"

        x_end, y_end = bwd_pass[0,0],bwd_pass[0,1]

        _arrow_dict["x"] ,_arrow_dict["y"] = x_end, y_end
        _arrow_dict["ax"],_arrow_dict["ay"] = 0, 0

        arrow2= go.layout.Annotation(dict(
                    _arrow_dict
                    ))

        layout2 = go.Layout(annotations = [arrow1,arrow2])

        if k<N:

            frame = go.Frame(data=[go.Scatter(name = "forward pass/state",
                x = fwd_pass[:k,0],
                y = fwd_pass[:k,1],
                mode="lines")],
                   )
        elif k>=N:
            frame = go.Frame(data=[go.Scatter(name = "forward pass/state",
                        x = fwd_pass[:N,0],
                        y = fwd_pass[:N,1],
                        mode="lines"),

                          go.Scatter( name = "backward pass/adjoint state",
                            x = bwd_pass[:k-N,0],
                            y = bwd_pass[:k-N,1],
                            mode="lines")],
                            layout = layout2)

        frames.append(frame)



    fig.frames = frames

    return fig