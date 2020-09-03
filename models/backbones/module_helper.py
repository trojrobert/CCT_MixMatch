import torch.nn as nn

class ModuleHelper():

    @staticmethod
    def BatchNorm2d(norm_type=None, ret_cls=False):
        if norm_type == 'batchnorm':
            return nn.BatchNorm2d
        elif norm_type == 'encsync_batchnorm':
            from encoding.nn import BatchNorm2d
            return BatchNorm2d

        elif norm_type == 'instancenorm':
            return nn.InstanceNorm2d
        else:
            raise ValueError('Not support BN type: {}.'.format(norm_type))


    def load_model(model, pretrained=None, all_match=True, map_location='cpu'):

        if pretrained is None: 
            return model

        if not os.path.exists(pretrained): 
            print('{}  not exist.'. format(pretrained))
            return model 
        
        print('Loading pretrained model: {}'.format(pretrained))

        if all_match: 

            pretrained_dict = torch.load(pretrained, map_location=map_location)
            model_dict = model.state_dict()

            load_dict = dict()

            for k, v in pretrained_dict.items():
                if 'prefix.{}'.format(k) in model_dict:
                    load_dict['prefix.{}'.format(k)] = v

                else: 
                    load_dict[k] = v
                model.load_state_dict(load_dict)

        else:

            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            print('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)

        return model