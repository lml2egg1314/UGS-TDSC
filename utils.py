def get_dctmtx(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if(i == 0):
                x = np.sqrt(1/n)
            else:
                x = np.sqrt(2/n)

            A[i][j] = x*np.cos(np.pi*(j+0.5)*i/n)

    return A

# a = get_dctmtx(8)
# print(a)

def get_qtable(QF):
    # standard quantization matrix
    D = np.array([[16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]], dtype=np.int64)

    if QF >= 50:
        s = 200 - 2*QF
    else:
        s = 5000/QF

    qtable = np.floor((50 + s*D)/100)

    return qtable


class Bidct(nn.Module):
    def __init__(self, qtable, mtx):
        super(Bidct, self).__init__()

        self.qtable = qtable
        self.mtx = mtx

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], int(x_shape[1]*x_shape[2]/8), 8, x_shape[3])
        x = x.permute(0,1,3,2)
        x = x.reshape(x_shape[0], int(x_shape[1]*x_shape[2]*x_shape[3]/(8*8)), 8, 8)
        x = x.permute(0,1,3,2)
        x = x * self.qtable
        x = torch.matmul(torch.t(self.mtx), x)
        x = torch.matmul(x, self.mtx)
        x = x.permute(0,1,3,2)
        x = x.reshape(x_shape[0], int(x_shape[1]*x_shape[2]/8), x_shape[2], 8)
        x = x.permute(0,1,3,2)
        x = x.reshape(x_shape) + 128

        return x

class Bdct(nn.Module):
    def __init__(self, qtable, mtx):
        super(Bdct, self).__init__()

        self.qtable = qtable
        self.mtx = mtx

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], int(x_shape[1] * x_shape[2] / 8), 8, x_shape[3])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x_shape[0], int(x_shape[1] * x_shape[2] * x_shape[3] / (8 * 8)), 8, 8)
        x = x.permute(0, 1, 3, 2)
        x = (x - 128) / self.qtable
        x = torch.matmul(self.mtx, x)
        x = torch.matmul(x, torch.t(self.mtx))
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x_shape[0], int(x_shape[1] * x_shape[2] / 8), x_shape[2], 8)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x_shape)

        return x

qtable = get_qtable(75)
qtable = torch.from_numpy(qtable).float().to(device)

mtx = get_dctmtx(8)
mtx = torch.from_numpy(mtx).float().to(device)

Bdct_layer = Bdct(qtable, mtx)
Bidct_layer = Bidct(qtable, mtx)