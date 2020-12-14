import numpy as np

def latent_dynamics_encoder(Env, params_to_encode = 'all'):    
    class DynamicsEncoding(Env):

        def reset(self, **kwargs):
            obs = super().reset(**kwargs)
            if params_to_encode == 'all':
                self.params_to_encode = self.params_dict
            else:
                self.params_to_encode = {k:v for k, v in self.params_dict.items() if k in params_to_encode}
            # print('reset: ', self.params_to_encode)
            return obs

    return DynamicsEncoding
