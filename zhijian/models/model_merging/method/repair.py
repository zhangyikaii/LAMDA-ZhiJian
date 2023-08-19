
import torch
import torch.nn as nn
from zhijian.data.base import prepare_vision_dataloader
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import scipy.optimize
from torch.cuda.amp import autocast

class REPAIR(object):
    def __init__(self, args, model_args, logger):
        self.args = args
        self.model_args = model_args
        self.logger = logger
        original_bs = self.args.batch_size
        self.args.batch_size = 500
        self.train_aug_loader, _, _ = prepare_vision_dataloader(self.args, self.model_args, self.logger)
        self.retrain_loader, _, _ = prepare_vision_dataloader(self.args, self.model_args, self.logger)
        self.args.batch_size = original_bs
        
        ...
    def core(self, fusion_model, merge_models_list):
        self.align_model(merge_models_list)
        mix_model = self.mix_weights(merge_models_list)
        wrap_a = self.correct_stat_with_repair(mix_model, merge_models_list)
        if self.args.retrain > 0:
            wrap_a = self.fit_merge(wrap_a)
        fusion_model = self.merge_model(fusion_model, wrap_a)

        return fusion_model


    def align_model(self, merge_models_list):
        feats1 = merge_models_list[1].features

        n = len(feats1)
        for i in range(n):
            if not isinstance(feats1[i], nn.Conv2d):
                continue
            
            # permute the outputs of the current conv layer
            assert isinstance(feats1[i+1], nn.ReLU)
            perm_map = self.get_layer_perm(self.subnet(merge_models_list[0], i+2), self.subnet(merge_models_list[1], i+2))
            self.permute_output(perm_map, feats1[i])
            
            # look for the next conv layer, whose inputs should be permuted the same way
            next_layer = None
            for j in range(i+1, n):
                if isinstance(feats1[j], nn.Conv2d):
                    next_layer = feats1[j]
                    break
            if next_layer is None:
                next_layer = merge_models_list[1].classifier
            self.permute_input(perm_map, next_layer)

    def correct_stat_with_repair(self, mix_model, merge_model_list):
        ## Calculate all neuronal statistics in the endpoint networks
        wrap0 = self.make_tracked_net(merge_model_list[0])
        wrap1 = self.make_tracked_net(merge_model_list[1])
        self.reset_bn_stats(wrap0)
        self.reset_bn_stats(wrap1)

        wrap_a = self.make_repaired_net(mix_model)
        # Iterate through corresponding triples of (TrackLayer, TrackLayer, ResetLayer)
        # around conv layers in (model0, model1, model_a).
        for track0, track1, reset_a in zip(wrap0.modules(), wrap1.modules(), wrap_a.modules()): 
            if not isinstance(track0, TrackLayer):
                continue  
            assert (isinstance(track0, TrackLayer)
                    and isinstance(track1, TrackLayer)
                    and isinstance(reset_a, ResetLayer))

            # get neuronal statistics of original networks
            mu0, std0 = track0.get_stats()
            mu1, std1 = track1.get_stats()
            # set the goal neuronal statistics for the merged network 
            goal_mean = (1 - self.args.fusion_propotion) * mu0 + self.args.fusion_propotion * mu1
            goal_std = (1 - self.args.fusion_propotion) * std0 + self.args.fusion_propotion * std1
            reset_a.set_stats(goal_mean, goal_std)

        # Estimate mean/vars such that when added BNs are set to eval mode,
        # neuronal stats will be goal_mean and goal_std.
        self.reset_bn_stats(wrap_a)

        return wrap_a


    def merge_model(self, fusion_model, wrap_a):
        # fuse the rescaling+shift coefficients back into conv layers
        fusion_model = self.fuse_tracked_net(fusion_model, wrap_a)

        return fusion_model
    
    def get_optimizer(self, config, model_parameters):
        """
        Create an optimizer for a given model
        :param model_parameters: a list of parameters to be trained
        :return: Tuple (optimizer, scheduler)
        """
        self.logger.info('lr is ', config['optimizer_learning_rate'])
        if config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                model_parameters,
                lr=config['optimizer_learning_rate'],
                momentum=config['optimizer_momentum'],
                weight_decay=config['optimizer_weight_decay'],
            )
        else:
            raise ValueError('Unexpected value for optimizer')

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['optimizer_decay_at_epochs'],
            gamma=1.0/config['optimizer_decay_with_factor'],
        )

        return optimizer, scheduler

    def fit_merge(self, geometric_model):
        for param in geometric_model.parameters():
            param.requires_grad_(True)

        # Set the seed
        torch.manual_seed(self.args.retrain_seed)
        np.random.seed(self.args.retrain_seed)
        # torch.cuda.set_device(self.args.gpu)

        config = {}
        config['optimizer'] = self.args.retrain_optimizer
        config['optimizer_learning_rate'] = self.args.retrain_lr
        config['optimizer_momentum'] = self.args.retrain_momentum
        config['optimizer_weight_decay'] = self.args.retrain_wd
        config['optimizer_decay_at_epochs'] = self.args.retrain_decay_epoch
        config['optimizer_decay_with_factor'] = self.args.retrain_decay_factor

        optimizer, scheduler = self.get_optimizer(config, geometric_model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        self.logger.info("Number of epochs would be ", self.args.retrain)
        for epoch in range(self.args.retrain):
            self.logger.info('Epoch {:03d}'.format(epoch))

            # Enable training mode (automatic differentiation + batch norm)
            geometric_model.train()

            # Keep track of statistics during training
            mean_train_loss = Mean()

            # Update the optimizer's learning rate
            scheduler.step(epoch)

            for batch_x, batch_y in tqdm(self.retrain_loader):
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                # Compute gradients for the batch
                optimizer.zero_grad()
                prediction = geometric_model(batch_x)
                loss = criterion(prediction, batch_y)
                torch.nn.utils.clip_grad_norm_(parameters=geometric_model.parameters(), max_norm=10, norm_type=2)
                loss.backward()

                # Do an optimizer steps
                optimizer.step()

                # Store the statistics
                mean_train_loss.add(loss.item(), weight=len(batch_x))
            
            self.logger.info(f"Epoch {epoch + 1}: {mean_train_loss.value()}")

        return geometric_model

    # Given two networks net0, net1 which each output a feature map of shape NxCxWxH,
    # this will reshape both outputs to (N*W*H)xC
    # and then compute a CxC correlation matrix between the two
    def run_corr_matrix(self, net0, net1):
        n = len(self.train_aug_loader)
        with torch.no_grad():
            net0.cuda().eval()
            net1.cuda().eval()
            for i, (images, _) in enumerate(tqdm(self.train_aug_loader)):
                
                img_t = images.float().cuda()
                out0 = net0(img_t).double()
                out0 = out0.permute(0, 2, 3, 1).reshape(-1, out0.shape[1])
                out1 = net1(img_t).double()
                out1 = out1.permute(0, 2, 3, 1).reshape(-1, out1.shape[1])

                # save batchwise first+second moments and outer product
                mean0_b = out0.mean(dim=0)
                mean1_b = out1.mean(dim=0)
                sqmean0_b = out0.square().mean(dim=0)
                sqmean1_b = out1.square().mean(dim=0)
                outer_b = (out0.T @ out1) / out0.shape[0]
                if i == 0:
                    mean0 = torch.zeros_like(mean0_b)
                    mean1 = torch.zeros_like(mean1_b)
                    sqmean0 = torch.zeros_like(sqmean0_b)
                    sqmean1 = torch.zeros_like(sqmean1_b)
                    outer = torch.zeros_like(outer_b)
                mean0 += mean0_b / n
                mean1 += mean1_b / n
                sqmean0 += sqmean0_b / n
                sqmean1 += sqmean1_b / n
                outer += outer_b / n

        cov = outer - torch.outer(mean0, mean1)
        std0 = (sqmean0 - mean0**2).sqrt()
        std1 = (sqmean1 - mean1**2).sqrt()
        corr = cov / (torch.outer(std0, std1) + 1e-4)
        return corr

    def get_layer_perm1(self, corr_mtx):
        corr_mtx_a = corr_mtx.cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
        assert (row_ind == np.arange(len(corr_mtx_a))).all()
        perm_map = torch.tensor(col_ind).long()
        return perm_map

    # returns the channel-permutation to make layer1's activations most closely
    # match layer0's.
    def get_layer_perm(self, net0, net1):
        corr_mtx = self.run_corr_matrix(net0, net1)
        return self.get_layer_perm1(corr_mtx)
    
    # modifies the weight matrices of a convolution and batchnorm
    # layer given a permutation of the output channels
    def permute_output(self, perm_map, layer):
        pre_weights = [layer.weight,
                    layer.bias]
        for w in pre_weights:
            w.data = w[perm_map]

    # modifies the weight matrix of a layer for a given permutation of the input channels
    # works for both conv2d and linear
    def permute_input(self, perm_map, layer):
        w = layer.weight
        w.data = w[:, perm_map]
    
    def subnet(self, model, n_layers):
        return model.features[:n_layers]
    
    def mix_weights(self,  merge_model_list):
        mix_model = deepcopy(merge_model_list[0])
        sd_alpha = {k: (1 - self.args.fusion_propotion) * merge_model_list[0].state_dict()[k].cuda() + self.args.fusion_propotion * merge_model_list[1].state_dict()[k].cuda()
                    for k in merge_model_list[0].state_dict().keys()}
        mix_model.load_state_dict(sd_alpha)

        return mix_model

    # adds TrackLayers around every conv layer
    def make_tracked_net(self, net1):
        for i, layer in enumerate(net1.features):
            if isinstance(layer, nn.Conv2d):
                net1.features[i] = TrackLayer(layer)
        return net1.cuda().eval()

    # adds ResetLayers around every conv layer
    def make_repaired_net(self, net1):
        for i, layer in enumerate(net1.features):
            if isinstance(layer, nn.Conv2d):
                net1.features[i] = ResetLayer(layer)
        return net1.cuda().eval()
    
    # reset all tracked BN stats against training data
    def reset_bn_stats(self, model):
        # resetting stats to baseline first as below is necessary for stability
        for m in model.modules():
            if type(m) == nn.BatchNorm2d:
                m.momentum = None # use simple average
                m.reset_running_stats()
        model.train()
        with torch.no_grad(), autocast():
            for images, _ in self.train_aug_loader:
                output = model(images.cuda())

    def fuse_conv_bn(self, conv, bn):
        fused_conv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # set weights
        w_conv = conv.weight.clone()
        bn_std = (bn.eps + bn.running_var).sqrt()
        gamma = bn.weight / bn_std
        fused_conv.weight.data = (w_conv * gamma.reshape(-1, 1, 1, 1))

        # set bias
        beta = bn.bias + gamma * (-bn.running_mean + conv.bias)
        fused_conv.bias.data = beta
        
        return fused_conv

    def fuse_tracked_net(self, fusion_model, net):
        for i, rlayer in enumerate(net.features):
            if isinstance(rlayer, ResetLayer):
                fused_conv = self.fuse_conv_bn(rlayer.layer, rlayer.bn)
                fusion_model.features[i].load_state_dict(fused_conv.state_dict())
        fusion_model.classifier.load_state_dict(net.classifier.state_dict())
        return fusion_model
        


class TrackLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)
        
    def get_stats(self):
        return (self.bn.running_mean, self.bn.running_var.sqrt())
        
    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1

class ResetLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)
        
    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std
        
    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)
    
class Mean:
    """
    Running average of the values that are 'add'ed
    """
    def __init__(self, update_weight=1):
        """
        :param update_weight: 1 for normal, 2 for t-average
        """
        self.average = None
        self.counter = 0
        self.update_weight = update_weight

    def add(self, value, weight=1):
        """Add a value to the accumulator"""
        self.counter += weight
        if self.average is None:
            self.average = deepcopy(value)
        else:
            delta = value - self.average
            self.average += delta * self.update_weight * weight / (self.counter + self.update_weight - 1)
            if isinstance(self.average, torch.Tensor):
                self.average.detach()

    def value(self):
        """Access the current running average"""
        return self.average


class Max:
    """
    Keeps track of the max of all the values that are 'add'ed
    """
    def __init__(self):
        self.max = None

    def add(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max