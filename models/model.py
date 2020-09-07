import os  

from utils.losses import * 
from base import BaseModel
from models.encoder import Encoder

class CCT(BaseModel):

    def __init__(self, num_classes, conf, sup_loss=None,
                    cons_w_unsup=None, ignore_index=None,
                    testing=False, pretrained=True, use_weak_labels=False,
                    weakly_loss_w=0.4):

        if not testing: 
            assert (ignore_index is not None) and (sup_loss is not None) and (cons_w_unsup is not None)

        super(CCT, self).__init__()

        #check model mode, it can either be supervised or semi supervised 
        assert int(conf['supervised']) + int(conf['semi']) ==1, 'one mode only'

        if conf['supervised']:
            self.mode = 'supervised'
        else: 
            self.mode = 'semi'

        # unsupervised losses 
        self.ignore_index = ignore_index

        if conf['un_loss'] == 'MSE':
            self.unsuper_loss = softmax_mse_loss
        else:
            raise ValueError(f"Invalid supervised loss {conf['un_loss']}")

        self.unsup_loss_w = cons_w_unsup
        self.sup_loss_w = conf['supervised_w']

        self.softmax_temp = conf['softmax_temp']

        self.sup_loss = sup_loss 
        self.sup_type = conf['sup_loss']
        
        self.use_weak_lables = use_weak_labels
        self.weakly_loss_w = weakly_loss_w 

        self.aux_constraint = conf['aux_constraint']
        self.aux_constraint_w = conf['aux_constraint_w']

        self.confidence_th = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        # Create the model
        self.encoder = Encoder(pretrained=pretrained)  

        #save encoder structure to a file
        encoder_file_path = 'outputs/encoder_arch.txt'
        if not os.path.isfile(encoder_file_path):
            encoder_arch_file = open(encoder_file_path, 'w')
            encoder_arch_file.write(repr(self.encoder))
            encoder_arch_file.close()



