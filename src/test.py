from toy_experiment import *
import torch

# class Experiment:
#     def __init__(self, configs, Encoder, Decoder, Optimizer, AE, device='cpu') -> None:
#         self.configs = configs
#         self.Encoder = Encoder
#         self.Decoder = Decoder
#         self.Optimizer = Optimizer
#         self.AE = AE
#         self.device = device
    
#     def run(self, dataloader, epochs, lams, save_dir, eps=1e-4):
#         for lam in lams:
#             history = []
#             self.init_model(lam)
#             epoch = self.model.fit(
#                 dataloader=dataloader,
#                 epochs=epochs, # maybe convergence criterion is better
#                 history=history,
#                 eps=eps
#                 )
#             self.save_result(lam, history, epoch if epoch is not None else epochs, save_dir)
    
#     def save_result(self, lam, history, epochs, save_dir):
#         path = os.path.join(save_dir, '{:.0e}'.format(lam)) # model/man/amb/i/lam
#         os.makedirs(path, exist_ok=True)
#         self.model.save(path, epochs)
#         np.savetxt(os.path.join(path, self.model.name+'_history.csv'), np.asarray(history))
        
#     def init_model(self, lam):
#         self.model = self.AE(
#             self.configs, self.Encoder, self.Decoder, 
#             self.Optimizer, weights=[1., lam]
#             )
#         self.model.to(self.device)

# def generate_configs(data_dim, width, name, lr=1e-3):
#     configs = dict()
#     configs['encoder'] = {
#         'in_features': data_dim,
#         'out_features': min(128, data_dim),
#         'layer_width': width
#         }
#     configs['decoder'] = {
#         'in_features': min(128, data_dim),
#         'out_features': data_dim,
#         'layer_width': width
#         }
#     configs['optimizer'] = {'lr': lr}
#     configs['name'] = name
#     return configs

# def create_savedir(l, d, i):
#     dir = os.path.join('../saves/toy_new/non/')
#     os.makedirs(dir, exist_ok=True)
#     return dir

# def read_dataset(latent_dim, data_dim, i, device='cpu'):
#     path = '../data/toy_new/{}-manifold/{}-ambient/'.format(latent_dim, data_dim)
#     data_name = '{}-{}_{}.npy'.format(latent_dim, data_dim, i)
#     tensor = torch.as_tensor(np.load(os.path.join(path, data_name)), dtype=torch.float, device=device) # type:ignore
#     return TensorDataset(tensor)

def main_mp(ae_name, i, epochs=10000, batch=100, device='cpu', eps=None):
    ll = [1, 2, 4, 8, 16, 32][:4]
    dd = [2, 4, 8, 16, 32]
    lams = [0]
    ww = [[32]*4, [64]*4, [128]*4, [256]*4, [512]*4, [1024]*4][:4]

    for l, width in zip(ll, ww):
        for d in dd:
            dataset = read_dataset(l, d*l, i, device=device)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
            configs = generate_configs(d*l, width, ae_name)
            save_dir = create_savedir(l, d*l, i)

            experiment = Experiment(configs, MLP, SNMLP, Adam, ae_dict[ae_name], device=device) # SNMLP for spectral normalization
            experiment.run(dataloader=dataloader, epochs=epochs, lams=lams, save_dir=save_dir, eps=eps) # epochs to be modified

if __name__ == '__main__':

    for i in range(4):
        p = mp.Process(target=main_mp, args=('non', i, 10000, 100, 'cuda', 0))
        p.start()

    # l = 8
    # d = 32
    # i = 0
    # lams = [0] #10 ** np.linspace(-6, 0, 13)
    # width = [256]*4 # the low rec loss pertains to activation loss. Tanh seems better
    # device = 'cuda'

    # dataset = read_dataset(l, d*l, i, device=device) #TensorDataset(torch.rand(3000, 8*16, device=device))
    # print(dataset[:].std())
    # dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    # configs = generate_configs(d*l, width, 'non', lr=1e-4)
    # save_dir = create_savedir(l, d*l, i)

    # experiment = Experiment(configs, MLP, SNMLP, Adam, ae_dict['non'], device=device) # SNMLP for spectral normalization
    # experiment.run(dataloader=dataloader, epochs=20000, lams=lams, save_dir=save_dir, eps=5e-5) # epochs to be modified