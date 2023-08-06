
from typing import List
import torch
import os 
import pickle
import numpy as np
import ot
import copy
import math
from tqdm import tqdm

from zhijian.data.base import prepare_vision_dataloader
from zhijian.models.utils import set_seed
class OT(object):
    # def __init__(self, param1, param2, param3):
    def __init__(self, args, model_args, logger):
        self.args = args
        self.model_args = model_args
        self.logger = logger
        original_bs = self.args.batch_size
        self.args.batch_size = 1
        self.unit_batch_train_loader, _, _ = prepare_vision_dataloader(self.args, self.model_args, self.logger)
        self.args.batch_size = original_bs
        self.retrain_loader, _, _ = prepare_vision_dataloader(self.args, self.model_args, self.logger)

    def core(self, fusion_model, merge_models_list):
        activations = self.get_model_activations(merge_models_list)
        geometric_model = self.geometric_ensembling_modularized(fusion_model, merge_models_list, activations)
        if self.args.retrain > 0:
            geometric_model = self.fit_merge(geometric_model)

        fusion_model.load_state_dict(geometric_model.state_dict(), strict=False)

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

        set_seed(self.args.retrain_seed)

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
            
            self.logger.info(f"Epoch {epoch}: {mean_train_loss.value()}")

        return geometric_model
    

    def _reduce_layer_name(self, layer_name):
        # self.logger.info("layer0_name is ", layer0_name) It was features.0.weight
        # previous way assumed only one dot, so now I replace the stuff after last dot
        return layer_name.replace('.' + layer_name.split('.')[-1], '')

    def process_activations(self, activations, layer0_name, layer1_name):
        activations_0 = activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].squeeze(1)
        activations_1 = activations[1][layer1_name.replace('.' + layer1_name.split('.')[-1], '')].squeeze(1)

        # assert activations_0.shape == activations_1.shape
        self._check_activation_sizes(activations_0, activations_1)

        if self.args.same_model != -1:
            # sanity check when averaging the same model (with value being the model index)
            assert (activations_0 == activations_1).all()
            self.logger.info("Are the activations the same? ", (activations_0 == activations_1).all())

        if len(activations_0.shape) == 2:
            activations_0 = activations_0.t()
            activations_1 = activations_1.t()
        elif len(activations_0.shape) > 2:
            reorder_dim = [l for l in range(1, len(activations_0.shape))]
            reorder_dim.append(0)
            self.logger.info("reorder_dim is ", reorder_dim)
            activations_0 = activations_0.permute(*reorder_dim).contiguous()
            activations_1 = activations_1.permute(*reorder_dim).contiguous()

        return activations_0, activations_1
    
    def _check_activation_sizes(self, acts0, acts1):
        if self.args.width_ratio == 1:
            return acts0.shape == acts1.shape
        else:
            return acts0.shape[-1]/acts1.shape[-1] == self.args.width_ratio
        
    def get_activation_distance_stats(self, activations_0, activations_1, layer_name=""):
        if layer_name != "":
            self.logger.info("In layer {}: getting activation distance statistics".format(layer_name))
        M = self.cost_matrix(activations_0, activations_1) ** (1/2)
        mean_dists =  torch.mean(M, dim=-1)
        max_dists = torch.max(M, dim=-1)[0]
        min_dists = torch.min(M, dim=-1)[0]
        std_dists = torch.std(M, dim=-1)

        self.logger.info("Statistics of the distance from neurons of layer 1 (averaged across nodes of layer 0): \n")
        self.logger.info("Max : {}, Mean : {}, Min : {}, Std: {}".format(torch.mean(max_dists), torch.mean(mean_dists), torch.mean(min_dists), torch.mean(std_dists)))

    def cost_matrix(self, x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        return c
    
    def _get_layer_weights(self, layer_weight, is_conv):
        if is_conv:
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
        else:
            layer_weight_data = layer_weight.data

        return layer_weight_data
    
    def _get_updated_acts_v0(self, layer_shape, aligned_wt, model0_aligned_layers, networks, layer_names):
        '''
        Return the updated activations of the 0th model with respect to the other one.

        :param args:
        :param layer_shape:
        :param aligned_wt:
        :param model0_aligned_layers:
        :param networks:
        :param test_loader:
        :param layer_names:
        :return:
        '''
        if layer_shape != aligned_wt.shape:
            updated_aligned_wt = aligned_wt.view(layer_shape)
        else:
            updated_aligned_wt = aligned_wt

        updated_model0, _ = self.update_model(networks[0], model0_aligned_layers + [updated_aligned_wt])
        updated_activations = self.get_model_activations([updated_model0, networks[1]], selective=True)

        updated_activations_0, updated_activations_1 = self.process_activations(updated_activations,
                                                                        layer_names[0], layer_names[1])
        return updated_activations_0, updated_activations_1
    
    def update_model(self, model, new_params):
        updated_model = copy.deepcopy(model)
        if self.args.gpu != -1:
            updated_model = updated_model.cuda()

        layer_idx = 0
        model_state_dict = model.state_dict()

        self.logger.info("len of model_state_dict is ", len(model_state_dict.items()))
        self.logger.info("len of new_params is ", len(new_params))

        for key, value in model_state_dict.items():
            self.logger.info("updated parameters for layer ", key)
            model_state_dict[key] = new_params[layer_idx]
            layer_idx += 1
            if layer_idx == len(new_params):
                break


        updated_model.load_state_dict(model_state_dict)

        final_acc = None

        return updated_model, final_acc

    def _process_ground_metric_from_acts(self, is_conv, ground_metric_object, activations):
        self.logger.info("inside refactored")
        if is_conv:
            if not self.args.gromov:
                M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                                activations[1].view(activations[1].shape[0], -1))
            else:
                M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                                activations[0].view(activations[0].shape[0], -1))
                M1 = ground_metric_object.process(activations[1].view(activations[1].shape[0], -1),
                                                activations[1].view(activations[1].shape[0], -1))

            self.logger.info("# of ground metric features is ", (activations[0].view(activations[0].shape[0], -1)).shape[1])
        else:
            if not self.args.gromov:
                M0 = ground_metric_object.process(activations[0], activations[1])
            else:
                M0 = ground_metric_object.process(activations[0], activations[0])
                M1 = ground_metric_object.process(activations[1], activations[1])

        if self.args.gromov:
            return M0, M1
        else:
            return M0, None

    def _custom_sinkhorn(self, mu, nu, cpuM):
        if not self.args.unbalanced:
            if self.args.sinkhorn_type == 'normal':
                T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=self.args.reg)
            elif self.args.sinkhorn_type == 'stabilized':
                T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=self.args.reg)
            elif self.args.sinkhorn_type == 'epsilon':
                T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=self.args.reg)
            else:
                raise NotImplementedError
        else:
            T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=self.args.reg, reg_m=self.args.reg_m)
        return T

    def _sanity_check_tmap(self, T):
        if not math.isclose(np.sum(T), 1.0, abs_tol=1e-7):
            self.logger.info("Sum of transport map is ", np.sum(T))
            raise Exception('NAN inside Transport MAP. Most likely due to large ground metric values')

    def _get_current_layer_transport_map(self, mu, nu, M0, M1, idx, layer_shape, eps=1e-7, layer_name=None):

        if not self.args.gromov:
            cpuM = M0.data.cpu().numpy()
            if self.args.exact:
                T = ot.emd(mu, nu, cpuM)
            else:
                T = self._custom_sinkhorn(mu, nu, cpuM)

            if self.args.print_distances:
                ot_cost = np.multiply(T, cpuM).sum()
                self.logger.info(f'At layer idx {idx} and shape {layer_shape}, the OT cost is ', ot_cost)
                if layer_name is not None:
                    setattr(self.args, f'{layer_name}_layer_{idx}_cost', ot_cost)
                else:
                    setattr(self.args, f'layer_{idx}_cost', ot_cost)
        else:
            cpuM0 = M0.data.cpu().numpy()
            cpuM1 = M1.data.cpu().numpy()

            assert not self.args.exact
            T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=self.args.gromov_loss, epsilon=self.args.reg)

        if not self.args.unbalanced:
            self._sanity_check_tmap(T)

        if self.args.gpu != -1:
            T_var = torch.from_numpy(T).cuda().float()
        else:
            T_var = torch.from_numpy(T).float()

        if self.args.tmap_stats:
            self.logger.info(
            "Tmap stats (before correction) \n: For layer {}, frobenius norm from the joe's transport map is {}".format(
                layer_name, torch.norm(T_var - torch.ones_like(T_var) / torch.numel(T_var), p='fro')
            ))

        self.logger.info("shape of T_var is ", T_var.shape)
        self.logger.info("T_var before correction ", T_var)

        return T_var

    def geometric_ensembling_modularized(self, fusion_model, networks, activations=None):
        if self.args.geom_ensemble_type == 'wts':
            avg_aligned_layers = self.get_wassersteinized_layers_modularized(networks)
        elif self.args.geom_ensemble_type == 'acts':
            avg_aligned_layers = self.get_acts_wassersteinized_layers_modularized(networks, activations)
            
        return self.get_network_from_param_list(fusion_model, avg_aligned_layers)
    
    def get_network_from_param_list(self, fusion_model, param_list):
        self.logger.info("using independent method")

        layer_idx = 0
        model_state_dict = fusion_model.state_dict()

        self.logger.info("len of model_state_dict is ", len(model_state_dict.items()))
        self.logger.info("len of param_list is ", len(param_list))

        for key, value in model_state_dict.items():
            model_state_dict[key] = param_list[layer_idx]
            layer_idx += 1

        fusion_model.load_state_dict(model_state_dict)

        return fusion_model

    def get_wassersteinized_layers_modularized(self, networks, eps=1e-7):
        '''
        Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
        The 1st network is aligned with respect to the other via wasserstein distance.
        Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

        :param networks: list of networks
        :param activations: If not None, use it to build the activation histograms.
        Otherwise assumes uniform distribution over neurons in a layer.
        :return: list of layer weights 'wassersteinized'
        '''

        # simple_model_0, simple_model_1 = networks[0], networks[1]
        # simple_model_0 = get_trained_model(0, model='simplenet')
        # simple_model_1 = get_trained_model(1, model='simplenet')

        avg_aligned_layers = []
        # cumulative_T_var = None
        T_var = None
        # self.logger.info(list(networks[0].parameters()))
        previous_layer_shape = None
        ground_metric_object = GroundMetric(self.args, logger=self.logger)

        if self.args.eval_aligned:
            model0_aligned_layers = []

        if self.args.gpu==-1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(self.args.gpu))


        num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
        for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
                enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

            assert fc_layer0_weight.shape == fc_layer1_weight.shape
            self.logger.info("Previous layer shape is ", previous_layer_shape)
            previous_layer_shape = fc_layer1_weight.shape

            mu_cardinality = fc_layer0_weight.shape[0]
            nu_cardinality = fc_layer1_weight.shape[0]

            # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
            # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]

            layer_shape = fc_layer0_weight.shape
            if len(layer_shape) > 2:
                is_conv = True
                # For convolutional layers, it is (#out_channels, #in_channels, height, width)
                fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
                fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
            else:
                is_conv = False
                fc_layer0_weight_data = fc_layer0_weight.data
                fc_layer1_weight_data = fc_layer1_weight.data

            if idx == 0:
                if is_conv:
                    M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                    # M = cost_matrix(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                    #                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                else:
                    # self.logger.info("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                    M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
                    # M = cost_matrix(fc_layer0_weight, fc_layer1_weight)

                aligned_wt = fc_layer0_weight_data
            else:

                self.logger.info("shape of layer: model 0", fc_layer0_weight_data.shape)
                self.logger.info("shape of layer: model 1", fc_layer1_weight_data.shape)
                self.logger.info("shape of previous transport map", T_var.shape)

                # aligned_wt = None, this caches the tensor and causes OOM
                if is_conv:
                    T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                    aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1).cuda(), T_var_conv.cuda()).permute(1, 2, 0)

                    M = ground_metric_object.process(
                        aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                        fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                    )
                else:
                    if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                        # Handles the switch from convolutional layers to fc layers
                        fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                        aligned_wt = torch.bmm(
                            fc_layer0_unflattened,
                            T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                        ).permute(1, 2, 0)
                        aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                    else:
                        # self.logger.info("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                        aligned_wt = torch.matmul(fc_layer0_weight.data.cuda(), T_var.cuda())
                    # M = cost_matrix(aligned_wt, fc_layer1_weight)
                    M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                    self.logger.info("ground metric is ", M)
                if self.args.skip_last_layer and idx == (num_layers - 1):
                    self.logger.info("Simple averaging of last layer weights. NO transport map needs to be computed")
                    if self.args.fusion_propotion != 0.5:
                        avg_aligned_layers.append((1 - self.args.fusion_propotion) * aligned_wt +
                                            self.args.fusion_propotion * fc_layer1_weight)
                    else:
                        avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                    return avg_aligned_layers

            if self.args.importance is None or (idx == num_layers -1):
                mu = self.get_histogram(0, mu_cardinality, layer0_name)
                nu = self.get_histogram(1, nu_cardinality, layer1_name)
            else:
                # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
                mu = self._get_neuron_importance_histogram(fc_layer0_weight_data, is_conv)
                nu = self._get_neuron_importance_histogram(fc_layer1_weight_data, is_conv)
                self.logger.info(mu, nu)
                assert self.args.proper_marginals

            cpuM = M.data.cpu().numpy()
            if self.args.exact:
                T = ot.emd(mu, nu, cpuM)
            else:
                T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=self.args.reg)
            # T = ot.emd(mu, nu, log_cpuM)

            if self.args.gpu!=-1:
                T_var = torch.from_numpy(T).cuda().float()
            else:
                T_var = torch.from_numpy(T).float()

            # torch.set_printoptions(profile="full")
            self.logger.info("the transport map is ", T_var)
            # torch.set_printoptions(profile="default")

            if self.args.correction:
                if not self.args.proper_marginals:
                    # think of it as m x 1, scaling weights for m linear combinations of points in X
                    if self.args.gpu != -1:
                        # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu))  # T.t().shape[1] = T.shape[0]
                        marginals = torch.ones(T_var.shape[0]).cuda() / T_var.shape[0]
                    else:
                        # marginals = torch.mv(T_var.t(),
                        #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                        marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                    marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                    T_var = torch.matmul(T_var, marginals)
                else:
                    # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                    marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                    marginals = (1 / (marginals_beta + eps))
                    self.logger.info("shape of inverse marginals beta is ", marginals_beta.shape)
                    self.logger.info("inverse marginals beta is ", marginals_beta)

                    T_var = T_var * marginals
                    # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                    # this should all be ones, and number equal to number of neurons in 2nd model
                    self.logger.info(T_var.sum(dim=0))
                    # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

            if self.args.debug:
                if idx == (num_layers - 1):
                    self.logger.info("there goes the last transport map: \n ", T_var)
                else:
                    self.logger.info("there goes the transport map at layer {}: \n ".format(idx), T_var)

                self.logger.info("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

            self.logger.info("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
            self.logger.info("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
            setattr(self.args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

            if self.args.past_correction:
                self.logger.info("this is past correction for weight mode")
                self.logger.info("Shape of aligned wt is ", aligned_wt.shape)
                self.logger.info("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
                t_fc0_model = torch.matmul(T_var.t().cuda(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1).cuda())
            else:
                t_fc0_model = torch.matmul(T_var.t().cuda(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1).cuda())

            # Average the weights of aligned first layers
            if self.args.fusion_propotion != 0.5:
                geometric_fc = ((1-self.args.fusion_propotion) * t_fc0_model +
                                self.args.fusion_propotion * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                geometric_fc = (t_fc0_model.cuda() + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1).cuda())/2
            if is_conv and layer_shape != geometric_fc.shape:
                geometric_fc = geometric_fc.view(layer_shape)
            avg_aligned_layers.append(geometric_fc)

        return avg_aligned_layers
    
    def get_acts_wassersteinized_layers_modularized(self, networks, activations, eps=1e-7):
        '''
        Average based on the activation vector over data samples. Obtain the transport map,
        and then based on which align the nodes and average the weights!
        Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
        The 1st network is aligned with respect to the other via wasserstein distance.
        Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
        :param networks: list of networks
        :param activations: If not None, use it to build the activation histograms.
        Otherwise assumes uniform distribution over neurons in a layer.
        :return: list of layer weights 'wassersteinized'
        '''
        avg_aligned_layers = []
        T_var = None
        if self.args.handle_skips:
            skip_T_var = None
            skip_T_var_idx = -1
            residual_T_var = None
            residual_T_var_idx = -1

        marginals_beta = None
        # self.logger.info(list(networks[0].parameters()))
        previous_layer_shape = None
        num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
        ground_metric_object = GroundMetric(self.args, logger=self.logger)

        if self.args.update_acts or self.args.eval_aligned:
            model0_aligned_layers = []

        if self.args.gpu==-1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(self.args.gpu))

        networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))
        idx = 0
        incoming_layer_aligned = True # for input
        while idx < num_layers:
            ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
        # for idx,  in \
        #         enumerate(zip(network0_named_params, network1_named_params)):
            self.logger.info("\n--------------- At layer index {} ------------- \n ".format(idx))
            # layer shape is out x in
            # assert fc_layer0_weight.shape == fc_layer1_weight.shape
            # assert _check_layer_sizes(args, idx, fc_layer0_weight.shape, fc_layer1_weight.shape, num_layers)
            self.logger.info("Previous layer shape is ", previous_layer_shape)
            previous_layer_shape = fc_layer1_weight.shape

            # will have shape layer_size x act_num_samples
            layer0_name_reduced = self._reduce_layer_name(layer0_name)
            layer1_name_reduced = self._reduce_layer_name(layer1_name)

            self.logger.info("let's see the difference in layer names", layer0_name.replace('.' + layer0_name.split('.')[-1], ''), layer0_name_reduced)
            self.logger.info(activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].shape, 'shape of activations generally')
            # for conv layer I need to make the act_num_samples dimension the last one, but it has the intermediate dimensions for
            # height and width of channels, so that won't work.
            # So convert (num_samples, layer_size, ht, wt) -> (layer_size, ht, wt, num_samples)

            activations_0, activations_1 = self.process_activations(activations, layer0_name, layer1_name)

            # self.logger.info("activations for 1st model are ", activations_0)
            # self.logger.info("activations for 2nd model are ", activations_1)


            assert activations_0.shape[0] == fc_layer0_weight.shape[0]
            assert activations_1.shape[0] == fc_layer1_weight.shape[0]

            mu_cardinality = fc_layer0_weight.shape[0]
            nu_cardinality = fc_layer1_weight.shape[0]

            self.get_activation_distance_stats(activations_0, activations_1, layer0_name)

            layer0_shape = fc_layer0_weight.shape
            layer_shape = fc_layer1_weight.shape
            if len(layer_shape) > 2:
                is_conv = True
            else:
                is_conv = False

            fc_layer0_weight_data = self._get_layer_weights(fc_layer0_weight, is_conv)
            fc_layer1_weight_data = self._get_layer_weights(fc_layer1_weight, is_conv)

            if idx == 0 or incoming_layer_aligned:
                aligned_wt = fc_layer0_weight_data

            else:

                self.logger.info("shape of layer: model 0", fc_layer0_weight_data.shape)
                self.logger.info("shape of layer: model 1", fc_layer1_weight_data.shape)

                self.logger.info("shape of activations: model 0", activations_0.shape)
                self.logger.info("shape of activations: model 1", activations_1.shape)


                self.logger.info("shape of previous transport map", T_var.shape)

                # aligned_wt = None, this caches the tensor and causes OOM
                if is_conv:
                    if self.args.handle_skips:
                        assert len(layer0_shape) == 4
                        # save skip_level transport map if there is block ahead
                        if layer0_shape[1] != layer0_shape[0]:
                            if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                                self.logger.info(f'saved skip T_var at layer {idx} with shape {layer0_shape}')
                                skip_T_var = T_var.clone()
                                skip_T_var_idx = idx
                            else:
                                self.logger.info(
                                    f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                                # if it's a shortcut (128, 64, 1, 1)
                                residual_T_var = T_var.clone()
                                residual_T_var_idx = idx  # use this after the skip
                                T_var = skip_T_var
                            self.logger.info("shape of previous transport map now is", T_var.shape)
                        else:
                            if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                                T_var = (T_var + residual_T_var) / 2
                                self.logger.info("averaging multiple T_var's")
                            else:
                                self.logger.info("doing nothing for skips")
                    T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                    aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1).cuda(), T_var_conv.cuda()).permute(1, 2, 0)

                else:
                    if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                        # Handles the switch from convolutional layers to fc layers
                        # checks if the input has been reshaped
                        fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                        -1).permute(2, 0, 1)
                        aligned_wt = torch.bmm(
                            fc_layer0_unflattened,
                            T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                        ).permute(1, 2, 0)
                        aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                    else:
                        aligned_wt = torch.matmul(fc_layer0_weight.data.cuda(), T_var.cuda())


            #### Refactored ####

                if self.args.update_acts:
                    assert self.args.second_model_name is None
                    activations_0, activations_1 = self._get_updated_acts_v0(layer_shape, aligned_wt,
                                                                        model0_aligned_layers, networks, [layer0_name, layer1_name])

            if self.args.importance is None or (idx == num_layers -1):
                mu = self.get_histogram(0, mu_cardinality, layer0_name)
                nu = self.get_histogram(1, nu_cardinality, layer1_name)
            else:
                # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
                mu = self._get_neuron_importance_histogram(fc_layer0_weight_data, is_conv)
                nu = self._get_neuron_importance_histogram(fc_layer1_weight_data, is_conv)
                self.logger.info(mu, nu)
                assert self.args.proper_marginals

            if self.args.act_bug:
                # bug from before (didn't change the activation part)
                # only for reproducing results from previous version
                M0 = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:
                # debugged part
                self.logger.info("Refactored ground metric calc")
                M0, M1 = self._process_ground_metric_from_acts(is_conv, ground_metric_object,
                                                        [activations_0, activations_1])

                self.logger.info("# of ground metric features in 0 is  ", (activations_0.view(activations_0.shape[0], -1)).shape[1])
                self.logger.info("# of ground metric features in 1 is  ", (activations_1.view(activations_1.shape[0], -1)).shape[1])

            if self.args.debug and not self.args.gromov:
                # bug from before (didn't change the activation part)
                M_old = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
                self.logger.info("Frobenius norm of old (i.e. bug involving wts) and new are ",
                    torch.norm(M_old, 'fro'), torch.norm(M0, 'fro'))
                self.logger.info("Frobenius norm of difference between ground metric wrt old ",
                    torch.norm(M0 - M_old, 'fro') / torch.norm(M_old, 'fro'))

                self.logger.info("ground metric old (i.e. bug involving wts) is ", M_old)
                self.logger.info("ground metric new is ", M0)

            ####################

            if self.args.same_model!=-1:
                self.logger.info("Checking ground metric matrix in case of same models")
                if not self.args.gromov:
                    self.logger.info(M0)
                else:
                    self.logger.info(M0, M1)

            if self.args.skip_last_layer and idx == (num_layers - 1):

                if self.args.skip_last_layer_type == 'average':
                    self.logger.info("Simple averaging of last layer weights. NO transport map needs to be computed")
                    if self.args.fusion_propotion != 0.5:
                        self.logger.info("taking baby steps (even in skip) ! ")
                        avg_aligned_layers.append((1-self.args.fusion_propotion) * aligned_wt +
                                                self.args.fusion_propotion * fc_layer1_weight)
                    else:
                        avg_aligned_layers.append(((aligned_wt + fc_layer1_weight)/2))
                elif self.args.skip_last_layer_type == 'second':
                    self.logger.info("Just giving the weights of the second model. NO transport map needs to be computed")
                    avg_aligned_layers.append(fc_layer1_weight)

                return avg_aligned_layers

            self.logger.info("ground metric (m0) is ", M0)

            T_var = self._get_current_layer_transport_map(mu, nu, M0, M1, idx=idx, layer_shape=layer_shape, eps=eps, layer_name=layer0_name)

            T_var, marginals = self._compute_marginals(T_var, device, eps=eps)

            if self.args.debug:
                if idx == (num_layers - 1):
                    self.logger.info("there goes the last transport map: \n ", T_var)
                    self.logger.info("and before marginals it is ", T_var/marginals)
                else:
                    self.logger.info("there goes the transport map at layer {}: \n ".format(idx), T_var)

            self.logger.info("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
            self.logger.info("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
            setattr(self.args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

            if self.args.past_correction:
                self.logger.info("Shape of aligned wt is ", aligned_wt.shape)
                self.logger.info("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)

                t_fc0_model = torch.matmul(T_var.t().cuda(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1).cuda())
            else:
                t_fc0_model = torch.matmul(T_var.t().cuda(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1).cuda())

            # Average the weights of aligned first layers
            if self.args.fusion_propotion != 0.5:
                self.logger.info("taking baby steps! ")
                geometric_fc = (1 - self.args.fusion_propotion) * t_fc0_model + \
                            self.args.fusion_propotion * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            else:
                geometric_fc = (t_fc0_model.cuda() + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1).cuda()) / 2
            if is_conv and layer_shape != geometric_fc.shape:
                geometric_fc = geometric_fc.view(layer_shape)
            avg_aligned_layers.append(geometric_fc)


            # self.logger.info("The averaged parameters are :", geometric_fc)
            # self.logger.info("The model0 and model1 parameters were :", fc_layer0_weight.data, fc_layer1_weight.data)

            if self.args.update_acts or self.args.eval_aligned:
                assert self.args.second_model_name is None
                # the thing is that there might be conv layers or other more intricate layers
                # hence there is no point in having them here
                # so instead call the compute_activations script and pass it the model0 aligned layers
                # and also the aligned weight computed (which has been aligned via the prev T map, i.e. incoming edges).
                if is_conv and layer_shape != t_fc0_model.shape:
                    t_fc0_model = t_fc0_model.view(layer_shape)
                model0_aligned_layers.append(t_fc0_model)
                _, acc = self.update_model(networks[0], model0_aligned_layers)
                self.logger.info("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
                setattr(self.args, 'model0_aligned_acc_layer_{}'.format(str(idx)), acc)
                if idx == (num_layers - 1):
                    setattr(self.args, 'model0_aligned_acc', acc)

            incoming_layer_aligned = False
            next_aligned_wt_reshaped = None

            # remove cached variables to prevent out of memory
            activations_0 = None
            activations_1 = None
            mu = None
            nu = None
            fc_layer0_weight_data = None
            fc_layer1_weight_data = None
            M0 = None
            M1 = None
            cpuM = None

            idx += 1
        return avg_aligned_layers
    
    def _compute_marginals(self, T_var, device, eps=1e-7):
        if self.args.correction:
            if not self.args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                marginals = torch.ones(T_var.shape)
                if self.args.gpu != -1:
                    marginals = marginals.cuda()

                marginals = torch.matmul(T_var, marginals)
                marginals = 1 / (marginals + eps)
                self.logger.info("marginals are ", marginals)

                T_var = T_var * marginals

            else:
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                marginals = (1 / (marginals_beta + eps))
                self.logger.info("shape of inverse marginals beta is ", marginals_beta.shape)
                self.logger.info("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                self.logger.info(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

            self.logger.info("T_var after correction ", T_var)
            self.logger.info("T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(),
                                                                        T_var.std()))
        else:
            marginals = None

        return T_var, marginals
    
    # def _check_layer_sizes(self, layer_idx, shape1, shape2, num_layers):
    #     if self.args.width_ratio == 1:
    #         return shape1 == shape2
    #     else:
    #         if self.args.dataset == 'mnist':
    #             if layer_idx == 0:
    #                 return shape1[-1] == shape2[-1] and (shape1[0]/shape2[0]) == self.args.width_ratio
    #             elif layer_idx == (num_layers -1):
    #                 return (shape1[-1]/shape2[-1]) == self.args.width_ratio and shape1[0] == shape2[0]
    #             else:
    #                 ans = True
    #                 for ix in range(len(shape1)):
    #                     ans = ans and shape1[ix]/shape2[ix] == self.args.width_ratio
    #                 return ans
    #         elif self.args.dataset[0:7] == 'Cifar10':
    #             assert self.args.second_model_name is not None
    #             if layer_idx == 0 or layer_idx == (num_layers -1):
    #                 return shape1 == shape2
    #             else:
    #                 if (not args.reverse and layer_idx == (num_layers-2)) or (args.reverse and layer_idx == 1):
    #                     return (shape1[1] / shape2[1]) == args.width_ratio
    #                 else:
    #                     return (shape1[0]/shape2[0]) == args.width_ratio
    
    def get_histogram(self, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
        if activations is None:
            # returns a uniform measure
            if not self.args.unbalanced:
                self.logger.info("returns a uniform measure of cardinality: ", cardinality)
                return np.ones(cardinality)/cardinality
            else:
                return np.ones(cardinality)
        else:
            # return softmax over the activations raised to a temperature
            # layer_name is like 'fc1.weight', while activations only contains 'fc1'
            self.logger.info(activations[idx].keys())
            unnormalized_weights = activations[idx][layer_name.split('.')[0]]
            self.logger.info("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
            unnormalized_weights = unnormalized_weights.squeeze()
            assert unnormalized_weights.shape[0] == cardinality

            if return_numpy:
                if float64:
                    return torch.softmax(unnormalized_weights / self.args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                        np.float64)
                else:
                    return torch.softmax(unnormalized_weights / self.args.softmax_temperature, dim=0).data.cpu().numpy()
            else:
                return torch.softmax(unnormalized_weights / self.args.softmax_temperature, dim=0)

    def _get_neuron_importance_histogram(self, layer_weight, is_conv, eps=1e-9):
        self.logger.info('shape of layer_weight is ', layer_weight.shape)
        if is_conv:
            layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
        else:
            layer = layer_weight.cpu().numpy()
        
        if self.args.importance == 'l1':
            importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
                        np.float64) + eps
        elif self.args.importance == 'l2':
            importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
                        np.float64) + eps
        else:
            raise NotImplementedError

        if not self.args.unbalanced:
            importance_hist = (importance_hist/importance_hist.sum())
            self.logger.info('sum of importance hist is ', importance_hist.sum())
        # assert importance_hist.sum() == 1.0
        return importance_hist

    def get_model_activations(self, models, selective=False):

        assert self.args.act_num_samples > 0, "act_num_samples must be greater than 0"

        if self.args.activation_mode is None:
            activations = self.compute_activations_across_models(models)
        else:
            if selective and self.args.update_acts:
                activations = self.compute_selective_activation(models)
            else:
                activations = self.compute_activations_across_models_v1(models)

        return activations
    
    def compute_activations_across_models(self, models, dump_activations=False, dump_path=None):
        # hook that computes the mean activations across data samples
        def get_activation(activation, name):
            def hook(model, input, output):
                # self.logger.info("num of samples seen before", num_samples_processed)
                # self.logger.info("output is ", output.detach())
                if name not in activation:
                    activation[name] = output.detach()
                else:
                    # self.logger.info("previously at layer {}: {}".format(name, activation[name]))
                    activation[name] = (num_samples_processed * activation[name] + output.detach()) / (
                            num_samples_processed + 1)
                # self.logger.info("now at layer {}: {}".format(name, activation[name]))

            return hook

        # Prepare all the models
        activations = {}

        for idx, model in enumerate(models):

            # Initialize the activation dictionary for each model
            activations[idx] = {}

            # Set forward hooks for all layers inside a model
            for name, layer in model.named_modules():
                if name == '':
                    self.logger.info("excluded")
                    continue
                layer.register_forward_hook(get_activation(activations[idx], name))
                self.logger.info("set forward hook for layer named: ", name)

            # Set the model in train mode
            model.train()

        # Run the same data samples ('num_samples' many) across all the models
        num_samples_processed = 0
        for batch_idx, (data, target) in enumerate(self.unit_batch_train_loader):
            if self.args.gpu != -1:
                data = data.cuda()
            for idx, model in enumerate(models):
                model.cuda()
                model(data)
                model.cpu()
            num_samples_processed += 1
            if num_samples_processed == self.args.act_num_samples:
                break

        # Dump the activations for all models onto disk
        if dump_activations and dump_path is not None:
            for idx in range(len(models)):
                self.save_activations(idx, activations[idx], dump_path)

        # self.logger.info("these will be returned", activations)
        return activations
    

    def compute_selective_activation(self, models, dump_activations=False, dump_path=None):
        torch.manual_seed(self.args.activation_seed)

        # hook that computes the mean activations across data samples
        def get_activation(activation, name):
            def hook(model, input, output):
                # self.logger.info("num of samples seen before", num_samples_processed)
                # self.logger.info("output is ", output.detach())
                if name not in activation:
                    activation[name] = []

                activation[name].append(output.detach())

            return hook

        # Prepare all the models
        activations = {}
        forward_hooks = []

        assert self.args.disable_bias
        # handle below for bias later on!
        # self.logger.info("list of model named params ", list(models[0].named_parameters()))
        param_names = [tupl[0].replace('.weight', '') for tupl in models[0].named_parameters()]

        for idx, model in enumerate(models):

            # Initialize the activation dictionary for each model
            activations[idx] = {}
            layer_hooks = []
            # Set forward hooks for all layers inside a model
            for name, layer in model.named_modules():
                if name == '':
                    self.logger.info("excluded")
                elif self.args.dataset != 'mnist' and name not in param_names:
                    self.logger.info("this was continued, ", name)
                # elif name!= layer_name:
                #     self.logger.info("this layer was not needed, ", name)
                else:
                    layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
                    self.logger.info("set forward hook for layer named: ", name)

            forward_hooks.append(layer_hooks)
            # Set the model in train mode
            model.train()

        # Run the same data samples ('num_samples' many) across all the models
        num_samples_processed = 0
        for batch_idx, (data, target) in enumerate(self.unit_batch_train_loader):
            if num_samples_processed == self.args.num_samples:
                break
            if self.args.gpu != -1:
                data = data.cuda()
            for idx, model in enumerate(models):
                model(data)
            num_samples_processed += 1

        relu = torch.nn.ReLU()
        for idx in range(len(models)):
            for lnum, layer in enumerate(activations[idx]):
                self.logger.info('***********')
                activations[idx][layer] = torch.stack(activations[idx][layer])
                self.logger.info("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                torch.max(activations[idx][layer]),
                                                                torch.mean(activations[idx][layer])))
                # assert (activations[idx][layer] >= 0).all()

                if not self.args.prelu_acts and not lnum == (len(activations[idx]) - 1):
                    # self.logger.info("activation was ", activations[idx][layer])
                    self.logger.info("applying relu ---------------")
                    activations[idx][layer] = relu(activations[idx][layer])
                    # self.logger.info("activation now ", activations[idx][layer])
                    self.logger.info("after RELU: min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                                torch.max(activations[idx][layer]),
                                                                                torch.mean(activations[idx][layer])))
                if self.args.standardize_acts:
                    mean_acts = activations[idx][layer].mean(dim=0)
                    std_acts = activations[idx][layer].std(dim=0)
                    self.logger.info("shape of mean, std, and usual acts are: ", mean_acts.shape, std_acts.shape,
                        activations[idx][layer].shape)
                    activations[idx][layer] = (activations[idx][layer] - mean_acts) / (std_acts + 1e-9)
                elif self.args.center_acts:
                    mean_acts = activations[idx][layer].mean(dim=0)
                    self.logger.info("shape of mean and usual acts are: ", mean_acts.shape, activations[idx][layer].shape)
                    activations[idx][layer] = (activations[idx][layer] - mean_acts)

                self.logger.info("activations for idx {} at layer {} have the following shape ".format(idx, layer),
                    activations[idx][layer].shape)
                self.logger.info('-----------')
        # Dump the activations for all models onto disk
        if dump_activations and dump_path is not None:
            for idx in range(len(models)):
                self.save_activations(idx, activations[idx], dump_path)

        # Remove the hooks (as this was intefering with prediction ensembling)
        for idx in range(len(forward_hooks)):
            for hook in forward_hooks[idx]:
                hook.remove()

        # self.logger.info("selective activations returned are", activations)
        return activations
    
    def compute_activations(self, model, train_loader, num_samples):
        '''

        This method can be called from another python module. Example usage demonstrated here.
        Averages the activations across the 'num_samples' many inputs.

        :param model: takes in a pretrained model
        :param train_loader: the particular train loader
        :param num_samples: # of randomly selected training examples to average the activations over

        :return:  list of len: num_layers and each of them is a particular tensor of activations
        '''

        activation = {}
        num_samples_processed = 0

        # Define forward hook that averages the activations
        # over number of samples processed
        def get_activation(name):
            def hook(model, input, output):
                self.logger.info("num of samples seen before", num_samples_processed)
                # self.logger.info("output is ", output.detach())
                if name not in activation:
                    activation[name] = output.detach()
                else:
                    # self.logger.info("previously at layer {}: {}".format(name, activation[name]))
                    activation[name] = (num_samples_processed * activation[name] + output.detach()) / (num_samples_processed + 1)
                # self.logger.info("now at layer {}: {}".format(name, activation[name]))

            return hook

        model.train()

        # Set forward hooks for all the layers
        for name, layer in model.named_modules():
            if name == '':
                self.logger.info("excluded")
                continue
            layer.register_forward_hook(get_activation(name))
            self.logger.info("set forward hook for layer named: ", name)

        # Run over the samples in training set
        # datapoints= []
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.args.gpu != -1:
                data = data.cuda()
                # datapoints.append(data)
                model(data)
                num_samples_processed += 1
                if num_samples_processed == num_samples:
                    break
        return activation, None # datapoints


    def compute_activations_across_models_v1(self, models, dump_activations=False, dump_path=None):

        torch.manual_seed(self.args.activation_seed)

        # hook that computes the mean activations across data samples
        def get_activation(activation, name):
            def hook(model, input, output):
                if name not in activation:
                    activation[name] = []

                activation[name].append(output.detach())

            return hook

        # Prepare all the models
        activations = {}
        forward_hooks = []


        assert self.args.disable_bias
        # handle below for bias later on!
        # self.logger.info("list of model named params ", list(models[0].named_parameters()))
        param_names = [tupl[0].replace('.weight', '') for tupl in models[0].named_parameters()]
        for idx, model in enumerate(models):

            # Initialize the activation dictionary for each model
            activations[idx] = {}
            layer_hooks = []
            # Set forward hooks for all layers inside a model
            for name, layer in model.named_modules():
                if name == '':
                    self.logger.info("excluded")
                    continue
                elif self.args.dataset != 'mnist' and name not in param_names:
                    self.logger.info("this was continued, ", name)
                    continue
                layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
                self.logger.info("set forward hook for layer named: ", name)

            forward_hooks.append(layer_hooks)
            # Set the model in train mode
            model.train()

        # Run the same data samples ('num_samples' many) across all the models
        num_samples_processed = 0
        for batch_idx, (data, target) in enumerate(self.unit_batch_train_loader):
            # self.logger.info("data:", data)
            # assert 0
            if num_samples_processed == self.args.act_num_samples:
                break
            if self.args.gpu != -1:
                data = data.cuda()

            for idx, model in enumerate(models):
                model.cuda()
                model(data)
                model.cpu()

            num_samples_processed += 1

        relu = torch.nn.ReLU()
        maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        avgpool = torch.nn.AvgPool2d(kernel_size=1, stride=1)

        # Combine the activations generated across the number of samples to form importance scores
        # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'

        model_cfg = self.get_model_layers_cfg()

        for idx in range(len(models)):
            cfg_idx = 0
            for lnum, layer in enumerate(activations[idx]):
                self.logger.info('***********')
                activations[idx][layer] = torch.stack(activations[idx][layer])
                self.logger.info("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]), torch.max(activations[idx][layer]), torch.mean(activations[idx][layer])))
                # assert (activations[idx][layer] >= 0).all()

                if not self.args.prelu_acts and not lnum == (len(activations[idx])-1):
                    # self.logger.info("activation was ", activations[idx][layer])
                    self.logger.info("applying relu ---------------")
                    activations[idx][layer] = relu(activations[idx][layer])
                    # self.logger.info("activation now ", activations[idx][layer])
                    self.logger.info("after RELU: min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                    torch.max(activations[idx][layer]),

                                                                    torch.mean(activations[idx][layer])))
                    
                elif self.args.model == 'vgg11_nobias' and self.args.pool_acts and len(activations[idx][layer].shape)>3:

                    if self.args.pool_relu:
                        self.logger.info("applying relu ---------------")
                        activations[idx][layer] = relu(activations[idx][layer])

                    activations[idx][layer] = activations[idx][layer].squeeze(1)

                    # apply maxpool wherever the next thing in config list is 'M'
                    if (cfg_idx + 1) < len(model_cfg):
                        if model_cfg[cfg_idx+1] == 'M':
                            self.logger.info("applying maxpool ---------------")
                            activations[idx][layer] = maxpool(activations[idx][layer])
                            cfg_idx += 2
                        else:
                            cfg_idx += 1

                    # apply avgpool only for the last layer
                    if cfg_idx == len(model_cfg):
                        self.logger.info("applying avgpool ---------------")
                        activations[idx][layer] = avgpool(activations[idx][layer])

                    # unsqueeze back at axis 1
                    activations[idx][layer] = activations[idx][layer].unsqueeze(1)

                    self.logger.info("checking stats after pooling")
                    self.logger.info("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]),
                                                                    torch.max(activations[idx][layer]),
                                                                    torch.mean(activations[idx][layer])))
                
                if self.args.activation_mode == 'mean':
                    activations[idx][layer] = activations[idx][layer].mean(dim=0)
                elif self.args.activation_mode == 'std':
                    activations[idx][layer] = activations[idx][layer].std(dim=0)
                elif self.args.activation_mode == 'meanstd':
                    activations[idx][layer] = activations[idx][layer].mean(dim=0) * activations[idx][layer].std(dim=0)


                if self.args.standardize_acts:
                    mean_acts = activations[idx][layer].mean(dim=0)
                    std_acts = activations[idx][layer].std(dim=0)
                    self.logger.info("shape of mean, std, and usual acts are: ", mean_acts.shape, std_acts.shape, activations[idx][layer].shape)
                    activations[idx][layer] = (activations[idx][layer] - mean_acts)/(std_acts + 1e-9)
                elif self.args.center_acts:
                    mean_acts = activations[idx][layer].mean(dim=0)
                    self.logger.info("shape of mean and usual acts are: ", mean_acts.shape, activations[idx][layer].shape)
                    activations[idx][layer] = (activations[idx][layer] - mean_acts)
                elif self.args.normalize_acts:
                    self.logger.info("normalizing the activation vectors")
                    activations[idx][layer] = self.normalize_tensor(activations[idx][layer])
                    self.logger.info("min of act: {}, max: {}, mean: {}".format(torch.min(activations[idx][layer]), torch.max(activations[idx][layer]), torch.mean(activations[idx][layer])))

                self.logger.info("activations for idx {} at layer {} have the following shape ".format(idx, layer), activations[idx][layer].shape)


        # Dump the activations for all models onto disk
        if dump_activations and dump_path is not None:
            for idx in range(len(models)):
                self.save_activations(idx, activations[idx], dump_path)

        # Remove the hooks (as this was intefering with prediction ensembling)
        for idx in range(len(forward_hooks)):
            for hook in forward_hooks[idx]:
                hook.remove()

        return activations

    def save_activations(self, idx, activation, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        pkl_file = open(os.path.join(dump_path, 'model_{}_activations'.format(idx)), "wb")
        pickle.dump(activation, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def get_model_layers_cfg(self):
        self.logger.info('model is ', self.args.model)
        model_name = self.args.model.split(".")[-1]
        if model_name == 'mlpnet' or model_name =='encoder':
            return None
        elif model_name[0:3].lower()=='vgg':
            cfg_key = model_name[0:5].upper()
        elif model_name[0:6].lower() == 'resnet':
            return None
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG11_quad': [64, 'M', 512, 'M', 1024, 1024, 'M', 2048, 2048, 'M', 2048, 512, 'M'],
            'VGG11_doub': [64, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 512, 'M'],
            'VGG11_half': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        return cfg[cfg_key]

    def normalize_tensor(self, tens):
        tens_shape = tens.shape
        assert tens_shape[1] == 1
        tens = tens.view(tens_shape[0], 1, -1)
        norms = tens.norm(dim=-1)
        ntens = tens/norms.view(-1, 1, 1)
        ntens = ntens.view(tens_shape)
        return ntens


class GroundMetric:
    """
        Ground Metric object for Wasserstein computations:

    """

    def __init__(self, params, not_squared=False, logger=None):
        self.params = params
        self.ground_metric_type = params.ground_metric
        self.ground_metric_normalize = params.ground_metric_normalize
        self.reg = params.reg
        if hasattr(params, 'not_squared'):
            self.squared = not params.not_squared
        else:
            # so by default squared will be on!
            self.squared = not not_squared
        self.mem_eff = params.ground_metric_eff
        self.logger = logger

    def _clip(self, ground_metric_matrix):
        if self.params.debug:
            self.logger.info("before clipping", ground_metric_matrix.data)

        percent_clipped = (float((ground_metric_matrix >= self.reg * self.params.clip_max).long().sum().data) \
                           / ground_metric_matrix.numel()) * 100
        self.logger.info("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.params, 'percent_clipped', percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(min=self.reg * self.params.clip_min,
                                             max=self.reg * self.params.clip_max)
        if self.params.debug:
            self.logger.info("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            self.logger.info("Normalizing by max of ground metric and which is ", ground_metric_matrix.max())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            self.logger.info("Normalizing by median of ground metric and which is ", ground_metric_matrix.median())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            self.logger.info("Normalizing by mean of ground metric and which is ", ground_metric_matrix.mean())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (self.isnan(ground_metric_matrix).any())

    def isnan(self, x):
        return x != x

    def _cost_matrix_xy(self, x, y, p=2, squared = True):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            self.logger.info("dont leave off the squaring of the ground metric")
            c = c ** (1/2)
        if self.params.dist_normalize:
            assert NotImplementedError
        return c

    def _pairwise_distances(self, x, y=None, squared=True):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm.cuda() + y_norm.cuda() - 2.0 * torch.mm(x.cuda(), y_t.cuda())
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        if self.params.activation_histograms and self.params.dist_normalize:
            dist = dist/self.params.act_num_samples
            self.logger.info("Divide squared distances by the num samples")

        if not squared:
            self.logger.info("dont leave off the squaring of the ground metric")
            dist = dist ** (1/2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
                - coordinates, p=2, dim=2
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared = self.squared)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        self.logger.info("stats of vecs are: mean {}, min {}, max {}, std {}".format(
            norms.mean(), norms.min(), norms.max(), norms.std()
        ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        self.logger.info('Processing the coordinates to form ground_metric')
        if self.params.geom_ensemble_type == 'wts' and self.params.normalize_wts:
            self.logger.info("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        if self.params.debug:
            self.logger.info("coordinates is ", coordinates)
            if other_coordinates is not None:
                self.logger.info("other_coordinates is ", other_coordinates)
            self.logger.info("ground_metric_matrix is ", ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.debug:
            self.logger.info("ground_metric_matrix at the end is ", ground_metric_matrix)

        return ground_metric_matrix
    
import torch
from copy import deepcopy

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
