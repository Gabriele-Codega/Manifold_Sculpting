import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, origin_dim, latent_dim):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(origin_dim,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,latent_dim)
        )
    
    def forward(self,x):
        return self.nn(x)

class Decoder(nn.Module):
    def __init__(self, origin_dim, latent_dim):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(latent_dim,32),
            nn.Tanh(),
            nn.Linear(32,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,origin_dim)
        )

    def forward(self,x):
        return self.nn(x)

class AutoEncoder(nn.Module):
    def __init__(self, origin_dim, latent_dim) -> None:
        super().__init__()

        self.origin_dim = origin_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(origin_dim,latent_dim)
        self.decoder = Decoder(origin_dim,latent_dim)

    def forward(self,x):
        x = self.encoder(x)
        return self.decoder(x)
    

def train(model,loader,loss_function,optimiser,num_epochs,scheduler):
    mean_losses = []
    for epoch in range(num_epochs): 
        model.train()
        losses=[]

        for input,output in loader:
            optimiser.zero_grad()           
            output = model(input)
            loss = loss_function(output, input)
            losses.append(loss.cpu().data.item())
            loss.backward()
            optimiser.step()

        ml = np.mean(losses)
        scheduler.step(ml)
        mean_losses.append(ml)
        print(f'\rEpoch: {epoch}, l = {ml:.3e}',sep=' ',end= '',flush=True)

    return mean_losses

data = np.load('../data/swiss_roll.npy')
roll = data[:,:3].astype(np.float32)
roll = roll - np.mean(roll,axis=0)
phi = data[:,-1]

roll = torch.from_numpy(roll).float()

model = AutoEncoder(3,2)

# change mode to either 'train' or 'load'
MODE = 'load'

if MODE == 'train':
    dataset = torch.utils.data.TensorDataset(roll,roll)
    loader = torch.utils.data.DataLoader(dataset,batch_size=20,shuffle=True)
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,patience=10)

    losses = train(model, loader,loss,optim,1000,scheduler)

    torch.save(model,'autoencoder.pt')

elif MODE == 'load':
    model = torch.load('autoencoder.pt')


y = model(roll).detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(y[:,0],y[:,1],y[:,2],c = phi)
ax.set_title('Autoencoder')


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(losses)
# ax.set_title('loss')

plt.show()



