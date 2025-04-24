

from models.decoder.UperHead import UperHead,UperSegHead

from utils.tools.logger import Logger as Log
decoder_head = {

    "uper_head":UperHead,
    "uper_seghead":UperSegHead,
}


class BuildHead(object):
    def __init__(self,configer = None,**kwargs):
        self.configer = configer
    def build_head(self , name = None, **kwargs):
        if name is not None:
            arch_head = name #self.configer.get("network","head")["name"]
        else:
            arch_head = self.configer.get("network","head")["name"]


        if arch_head in decoder_head.keys():
            head = decoder_head[arch_head](**kwargs)
        else:
            #Log.error('Head {} is invalid.'.format(arch_head))
            raise Exception('Head {} is invalid.'.format(arch_head))

        return head