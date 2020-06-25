import torch
import torch.nn as nn
import torchvision
from deeplkt.models import resnet
from lib.nn import SynchronizedBatchNorm2d
from deeplkt.config import cfg
from os.path import join
import nvgpu
cfg.merge_from_file('deeplkt/config/config.yaml')


BatchNorm2d = SynchronizedBatchNorm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, img, segSize=None):

        # training
        # if segSize is None:
        #     if self.deep_sup_scale is not None: # use deep supervision technique
        #         (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
        #     else:
        #         pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

        #     loss = self.crit(pred, feed_dict['seg_label'])
        #     if self.deep_sup_scale is not None:
        #         loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
        #         loss = loss + loss_deepsup * self.deep_sup_scale

        #     acc = self.pixel_acc(pred, feed_dict['seg_label'])
        #     return loss, acc
        # # inference
        # else:
        pred = self.decoder(self.encoder(img, return_feature_maps=True), segSize=segSize)
        return pred


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def load_my_state_dict(model, pretrained_dict):
    
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        new_dict = {}
        for k, v in pretrained_dict.items():
            # print(k, v.shape)
            if not k.startswith('conv_last.4') and \
                not k.startswith('conv_last_deepsup'):
                # print(k)
                new_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict, strict=False)

    @staticmethod
    def build_encoder(fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        
        orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
        net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            # print(net_decoder.state_dict().keys())
            ModelBuilder.load_my_state_dict(net_decoder,
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )



class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]




# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)

        # if self.use_softmax:  # is True during inference
        x = nn.functional.interpolate(
            x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)
        x =  x[:, 0:1, :, :]
        # print("X = ", x.shape)
        return x


        # deep sup
        # conv4 = conv_out[-2]
        # _ = self.cbr_deepsup(conv4)
        # _ = self.dropout_deepsup(_)
        # _ = self.conv_last_deepsup(_)

        # x = nn.functional.log_softmax(x, dim=1)
        # _ = nn.functional.log_softmax(_, dim=1)

        # return (x, _)

class AttentionModule(nn.Module):
    def __init__(self, device):
        super(AttentionModule, self).__init__()
        self.device = device
        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

        # absolute paths of model weights
        cfg.MODEL.weights_encoder = join(
            cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = join(
            cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

        net_encoder = ModelBuilder.build_encoder(
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            # num_class=2,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        for i, param in enumerate(self.segmentation_module.encoder.parameters()):
            param.requires_grad = False
        for i, param in enumerate(self.segmentation_module.decoder.parameters()):
            if(param.shape[0] != 2):
                param.requires_grad = False
    
    def forward(self, img_resized_list, segSize=None):
        scores = torch.zeros((img_resized_list[0].shape[0], 1, 
                                segSize[0], segSize[1]), device=self.device)
        # print("Before segmentation = ", nvgpu.gpu_info()[0]['mem_used'])

        for img_batch in img_resized_list:        
            pred_tmp = self.segmentation_module(img_batch, segSize=segSize)
            # print("After segmentation = ", nvgpu.gpu_info()[0]['mem_used'])
            del img_batch
            # print("Size = ", pred_tmp.size())
            # from IPython import embed;embed()
            scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)
        return scores



