import numpy as np

def latent_dynamics_provider(Env, params_to_attach = 'all'):    
    class DynamicsParams(Env):

        def reset(self, **kwargs):
            obs = super().reset(**kwargs)
            if params_to_attach == 'all':
                self.params_to_attach = self.params_dict
            else:
                self.params_to_attach = {k:v for k, v in self.params_dict.items() if k in params_to_attach}
            # print('reset: ', self.params_to_encode)
            info={}
            info['dynamics_params'] = self.params_to_attach.values()
            return obs, info

        def step(self, action):
            obs, reward, done, info = super().step(action)
            info['dynamics_params'] = self.params_to_attach.values()
            return obs, reward, done, info

    return DynamicsParams
