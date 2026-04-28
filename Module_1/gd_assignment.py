import torch
import matplotlib.pyplot as plt

def setup():
    """Set up the plotting style, size, and random seed for reproducibility."""
    plt.style.use('ggplot')
    torch.manual_seed(0)
    plt.rcParams["figure.figsize"] = (15, 8)

def gradient_wrt_m_and_c(inputs, labels, m, c, k):
    """
    Compute the gradient of the loss function w.r.t m and c for a single data point.
    inputs (torch.tensor): input (X)
    labels (torch.tensor): label (Y)
    m (float): slope of the line
    c (float): vertical intercept of line
    k (torch.tensor, dtype=int): random index of data points
    """
    # gradient w.r.t to m is g_m 
    # gradient w.r.t to c is g_c
    x = inputs[k]
    y = labels[k]
    
    num_samples = len(inputs)
    
    error = (m * x + c) - y
    g_m = 2 * torch.sum(error * x)
    g_c = 2 * torch.sum(error)

    return g_m, g_c

def update_m_and_c(m, c, g_m, g_c, lr):
    """
    Update the parameters m and c using the computed gradients and learning rate.
    """
    # update m and c parameters
    # store updated value of m is updated_m variable
    # store updated value of c is updated_c variable
    ###
    updated_m = m - lr * g_m
    updated_c = c - lr * g_c
    ###
    return updated_m, updated_c

def main():
    # Generating y = mx + c + random noise
    num_data = 1000

    # True values of m and c
    m_line = 3.3
    c_line = 5.3

    # input (Generate random data between [-5,5])
    x = 10 * torch.rand(num_data) - 5

    # Output (Generate data assuming y = mx + c + noise)
    y_label = m_line * x + c_line + torch.randn_like(x)
    y = m_line * x + c_line

    # Plot the generated data points 
    plt.plot(x, y_label, '.', color='g', label="Data points")
    plt.plot(x, y, color='b', label='y = mx + c', linewidth=3)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()

    X = torch.tensor([-0.0374,  2.6822, -4.1152])
    Y = torch.tensor([ 5.1765, 14.1513, -8.2802])
    m = 2
    c = 3
    k = torch.tensor([0, 2])

    gm, gc = gradient_wrt_m_and_c(X, Y, m, c, k)

    print('Gradient of m : {0:.2f}'.format(gm))
    print('Gradient of c : {0:.2f}'.format(gc))

    m = 2
    c = 3
    g_m = -24.93
    g_c = 1.60
    lr = 0.001
    m, c = update_m_and_c(m, c, g_m, g_c, lr)

    print('Updated m: {0:.2f}'.format(m))
    print('Updated c: {0:.2f}'.format(c))


if __name__ == "__main__":
    setup()
    main()