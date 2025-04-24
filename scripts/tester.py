from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.tools.logger import Logger as Log



import torch.nn as nn
from utils.distributed import get_world_size, get_rank, is_distributed
import torch
import time
import os
import torch.nn.functional as F
import torch.distributed as dist

from utils.tools.average_meter import AverageMeter


from models import clip
from models.sam.build_sam import build_sam_vit_b
from lib.loss.loss_manager import LossManager
from lib.vis.seg_visualizer import SegVisualizer
from dataset.data_loader import DataLoader
from scripts.tools.optim_scheduler import OptimScheduler

from scripts.tools.module_runner import ModuleRunner
from scripts.tools.evaluator import get_evaluator
from Inference.pre_process.predict_manager import PredictManager
from Inference.post_process.post_manager import PostManager
class Tester(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.evaluator = get_evaluator(configer, self)
        self.predict_manager = PredictManager(configer)
        self.post_manager = PostManager(configer)
        self.seg_visualizer = SegVisualizer(configer)

        self.train_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        self.visualizer = self.configer.get("eval", "visualizer")

        ######
        # build model and set hyperparameters######
        self.__init_trainer()

    def build_clip(self):
        clip_model, _ = clip.load("./pretrained/clip_checkpoints/ViT-B-16.pt", "cuda")
        return clip_model.eval()

    def build_sam(self):
        # load sam
        sam = build_sam_vit_b(checkpoint=r"./pretrained/sam_checkpoints/sam_vit_b.pth").eval()
        # load clip
        return sam

    def conduct_classnames(self):
        xml_data = self.configer.get("data", "classnames")
        from xml.dom.minidom import parse
        dom_tree = parse(xml_data)
        # type_elements = dom_tree.getElementsByTagName('type')
        # 获取所有的 <type> 元素
        type_elements = dom_tree.getElementsByTagName('type')

        types = []
        # 遍历每个 <type> 元素，并打印其文本内容
        for type_element in type_elements:
            # 获取元素的文本内容，并去除两端的空白字符
            types.append(type_element.firstChild.data.strip())
            # print(type_element.firstChild.data.strip())

        return types

    def __init_trainer(self):

        from models.SamClipCD import SamClipCD, ClipCD

        self.use_fp16 = self.configer.get("train", "fp16")

        Log.info("use fp16={} for trainning".format(self.configer.get("train", "fp16")))

        clip_model = self.build_clip()
        sam_model = self.build_sam()

        class_names = self.conduct_classnames()
        self.model = SamClipCD(
            self.configer,
            clip_model=clip_model,
            sam_model=sam_model,
            context_length=10,
            class_names=class_names,
            token_embed_dim=512,
            text_dim=512,
            prompt=self.configer.get("network", "prompt")
        )
        # self.model = ClipCD(
        #     self.configer,
        #     clip_model,
        #     context_length=10,
        #     class_names=class_names,
        #     token_embed_dim=512,
        #     text_dim=512
        # )

        self.module_runner.load_net(self.model)
        self.model.eval()




    def test(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        import time
        #self.seg_net.eval()


        '''   params for seg
            --image "D:\dengkai\data\变化检测测试数据\1209数据集7组\1同分辨率同坐标系\hou\GF220191.tif"
            --minx "801495.8202514648"
            --miny "2696516.4037475586"
            --maxx "805557.4202514648"
            --maxy "2700487.6037475588"
            --output_image "D:\dengkai\data\变化检测测试数据\1209数据集7组\1同分辨率同坐标系\out\0_GF220191_BJ220181\GF220191_preout.tif"
            --win_x "20000"
            --win_y "20000"
            --stride "20000"
        '''
        '''   
        params for seg_chg
        --left_image "D:\dengkai\data\变化检测测试数据\1209数据集7组\1同分辨率同坐标系\qian\BJ220181.tif" 
        --right_image "D:\dengkai\data\变化检测测试数据\1209数据集7组\1同分辨率同坐标系\hou\GF220191.tif" 
        --output_image "D:\dengkai\data\变化检测测试数据\1209数据集7组\1同分辨率同坐标系\out\BJ220181_GF220191\cd_corretion.tif" 
        --miny "2696516.4037475586" 
        --minx "801495.8202514648" 
        --maxx "805557.4202514648" 
        --maxy "2700487.6037475588"
    '''



        # self.inference = self.model_manager.semantic_segmentor()
        # kargs = {}
        # kargs['minx'] = self.configer.get('test','minx')
        # kargs['maxx'] = self.configer.get('test','maxx')
        # kargs['miny'] = self.configer.get('test','miny')
        # kargs['maxy'] = self.configer.get('test','maxy')
        # kargs['stride'] = self.configer.get('test','stride')
        # kargs['num_cls'] = self.configer.get('test','num_cls')
        # kargs['binv'] = self.configer.get('test','binv')
        # kargs['scales'] = self.configer.get('test','scales')
        # kargs['method_cd'] = self.configer.get('test','method_cd')
        # kargs['shpfile'] = self.configer.get('test','shpfile')
        # transxml = self.configer.get('config','transxml')
        # if not os.path.exists(transxml):
        #     transxml  = None
        # kargs['transxml'] = transxml
        # kargs['thr'] = self.configer.get('test','thr')
        # kargs['cr'] = self.configer.get('test','cr')
        # kargs['build_thr'] = self.configer.get('test','build_thr')
        # kargs['minsz'] = self.configer.get('test','minsz')
        # if self.configer.exists('test','weights'):
        #     weights = self.configer.get('test','weights')
        #     weights = np.array(weights)
        #     kargs['weights'] = weights


        minx = self.configer.get('test','minx')
        maxx = self.configer.get('test','maxx')
        miny = self.configer.get('test','miny')
        maxy = self.configer.get('test','maxy')
        winx = self.configer.get('test','winx')
        winy = self.configer.get('test','winy')
        stride = self.configer.get('test','stride')
        binv = self.configer.get('test','binv')
        scales = self.configer.get('test','scales')
        method_cd = self.configer.get('test','method_cd')
        shpfile = self.configer.get('test','shpfile')
        transxml = self.configer.get('config','transxml')
        num_cls = self.configer.get('data','num_classes')


        if "cd" in self.configer.get("test","method") :
            #
            # import json, cv2
            # import  numpy as np
            # from tqdm import  tqdm
            # data_json = self.configer.get("data", "valtxt")
            # with open(data_json, 'r', encoding="utf-8") as file:
            #     self.annotations = json.load(file)
            #
            # self.images = self.annotations['images']
            # out_dir = r"D:\dengkai\code\clip_sam_cd\vis\results\seg\test_maincd"
            # for i, image in tqdm(enumerate(self.images)):
            #     oldimage_path = image["oldpath"].replace("\\", "/")
            #     newimage_path = image["newpath"].replace("\\", "/")
            #     oldimg = cv2.imdecode(np.fromfile(oldimage_path, dtype=np.uint8), -1)
            #     newimg = cv2.imdecode(np.fromfile(newimage_path, dtype=np.uint8), -1)
            #
            #     # oldimg = cv2.cvtColor(oldimg, cv2.COLOR_BGR2RGB)
            #     # newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
            #     output_img_left = os.path.join(out_dir, f"left_{i:06d}.tif")
            #     output_img_right = os.path.join(out_dir, f"right_{i:06d}.tif")
            #     cv2.imwrite(output_img_left, oldimg )
            #     cv2.imwrite(output_img_right,newimg )
            #     output_img_file = os.path.join(out_dir,  f"{i:06d}.tif")
            #     self.predict_manager.predict_segmentor(
            #         configer=self.configer,
            #         input_src_file_left=oldimage_path,
            #         input_src_file_right=newimage_path,
            #         output_img_file=output_img_file,
            #         model=self.model,
            #         transxml=transxml,
            #         shpfile=shpfile,
            #         minx=minx, miny=miny, maxx=maxx, maxy=maxy,
            #         winx=winx, winy=winy, stride=stride,
            #         binary=0, scales=scales, method_cd=method_cd,
            #         num_classes=num_cls
            #     )
            # return
            input_src_file_left = self.configer.get('test','left_image')
            input_src_file_right = self.configer.get('test','right_image')
            output_img_file = self.configer.get('test','output_image')
            Log.info("predict process")
            t1 = time.time()
            self.predict_manager.predict_segmentor(
                configer  = self.configer,
                input_src_file_left = input_src_file_left,
                input_src_file_right = input_src_file_right,
                output_img_file = output_img_file,
                model = self.model,
                transxml = transxml,
                shpfile = shpfile,
                minx = minx,miny = miny,maxx = maxx,maxy = maxy,
                winx = winx,winy = winy,stride = stride,
                binary=binv,scales=scales,method_cd=method_cd,
                num_classes= num_cls
            )

            t2 = time.time()
            Log.info("time used {}s".format(t2 -t1))

            Log.info("post process")
            t1 = time.time()
            if self.configer.exists("post"):
                post_method = self.configer.get("post","method")
                if post_method == 'none':
                    Log.info("post method is {}".format(post_method))
                    return
                post_kwargs = self.configer.get("post","param")
                post_kwargs['shpfile'] = shpfile
                output_bin_file = os.path.splitext(output_img_file)[0] +  "_binary.tif"
                self.post_manager.post_process(
                    input_src_file_left = input_src_file_left,
                    input_src_file_right = input_src_file_right,
                    output_img_file = output_img_file,
                    output_bin_file = output_bin_file,
                    **post_kwargs
                )
            t2 = time.time()
            Log.info("time used {}s".format(t2 -t1))
        else:
            Log.error("{} is not valid".format(self.configer.get("method")))