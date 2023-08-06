from zhijian.models.backbone.base import prepare_model, prepare_pretrained
from zhijian.trainers.finetune import Trainer as Base_Trainer
from zhijian.models.model_merging.base import prepare_merging_method
from zhijian.models.configs.base import FB_RESNET_MEAN, FB_RESNET_STD

def prepare_specific_trainer_parser(parser):
    parser.add_argument('--merging-mode', type=str, default=None)
    parser.add_argument('--soup-mode', type=str, default=None)
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('--retrain-seed', type=int, default=1, action='store',
                        help='if reseed computations again in retrain')
    parser.add_argument('--retrain-optimizer', type=str, default='SGD', choices=['SGD'], help='which optimizer to use')
    parser.add_argument('--retrain-lr', type=float, default=0.001, help='retrain lr')
    parser.add_argument('--retrain-momentum', type=float, default=0.9, help='retrain momentum')
    parser.add_argument('--retrain-wd', type=float, default=0, help='retrain weight decay')
    parser.add_argument('--retrain-decay-epoch', type=int, default=50, help='retrain steplr epoch')
    parser.add_argument('--retrain-decay-factor', type=float, default=2, help='retrain steplr factor')

    parser.add_argument('--fusion-propotion', type=float, default=0.5, action='store', help='rate of adjustment towards the second model')

    # OT Fusion
    parser.add_argument('--reg', default=1e-2, type=float, help='regularization strength for sinkhorn (default: 1e-2)')
    parser.add_argument('--geom-ensemble-type', type=str, default='acts', choices=['wts', 'acts'],
                        help='Ensemble based on weights (wts) or activations (acts).')
    parser.add_argument('--act-num-samples', type=float, default=100)
    parser.add_argument('--activation-mode', type=str, default=None, choices=['mean', 'std', 'meanstd', 'raw'],
                        help='mode that chooses how the importance of a neuron is calculated.')
    parser.add_argument('--update-acts', action='store_true', help='update acts during the alignment of model0')
    parser.add_argument('--prelu-acts', action='store_true',
                        help='do activation based alignment based on pre-relu acts')
    parser.add_argument('--activation-seed', type=int, default=42, action='store', help='seed for computing activations')
    parser.add_argument('--disable-bias', action='store_true',
                        help='no bias at all in fc or conv layers')
    parser.add_argument('--normalize-acts', action='store_true',
                        help='normalize the vector of activations')
    parser.add_argument('--normalize-wts', action='store_true',
                        help='normalize the vector of weights')
    parser.add_argument('--standardize-acts', action='store_true',
                        help='subtract mean and divide by standard deviation across the samples for use in act based alignment')
    parser.add_argument('--center-acts', action='store_true',
                        help='subtract mean only across the samples for use in act based alignment')
    parser.add_argument('--pool-acts', action='store_true',
                        help='do activation based alignment based on pooling acts')
    parser.add_argument('--pool-relu', action='store_true',
                        help='do relu first before pooling acts')
    parser.add_argument('--eval-aligned', action='store_true',
                        help='evaluate the accuracy of the aligned model 0')
    parser.add_argument('--skip-last-layer', action='store_true', help='skip the last layer in calculating optimal transport')
    parser.add_argument('--ensemble-step', type=float, default=0.5, action='store', help='rate of adjustment towards the second model')
    parser.add_argument('--importance', type=str, default=None, action='store',
                        help='importance measure to use for building probab mass! (options, l1, l2, l11, l12)')
    parser.add_argument('--unbalanced', action='store_true', help='use unbalanced OT')
    parser.add_argument('--softmax-temperature', default=1, type=float, help='softmax temperature for activation weights (default: 1)')
    parser.add_argument('--proper-marginals', action='store_true', help='consider the marginals of transport map properly')
    parser.add_argument('--exact', action='store_true', help='compute exact optimal transport')
    # OT Fusion
    parser.add_argument('--correction', action='store_true', help='scaling correction for OT')
    parser.add_argument('--debug', action='store_true', help='print debug statements')
    parser.add_argument('--past-correction', action='store_true', help='use the current weights aligned by multiplying with past transport map')
    parser.add_argument('--handle-skips', action='store_true', help='handle shortcut skips in resnet which decrease dimension')
    parser.add_argument('--width-ratio', type=float, default=1, action='store',
                        help='ratio of the widths of the hidden layers between the two models')
    parser.add_argument('--second-model-name', type=str, default=None, action='store', help='name of second model!')
    parser.add_argument('--same-model', action='store', type=int, default=-1, help='Index of the same model to average with itself')
    parser.add_argument('--act-bug', action='store_true',
                        help='simulate the bug in ground metric calc for act based averaging')
    parser.add_argument('--gromov', action='store_true', help='use gromov wasserstein distance and barycenters')
    parser.add_argument('--skip-last-layer-type', type=str, default='average', choices=['second', 'average'],
                        help='how to average the parameters for the last layer')
    parser.add_argument('--sinkhorn-type', type=str, default='normal', choices=['normal', 'stabilized', 'epsilon', 'gpu'],
                        help='Type of sinkhorn algorithm to consider.')
    parser.add_argument('--reg-m', default=1e-3, type=float, help='regularization strength for marginals in unbalanced sinkhorn (default: 1e-3)')
    parser.add_argument('--print-distances', action='store_true', help='print OT distances for every layer')
    parser.add_argument('--gromov-loss', type=str, default='square_loss', action='store',
                        choices=['square_loss', 'kl_loss'], help="choice of loss function for gromov wasserstein computations")
    parser.add_argument('--tmap-stats', action='store_true', help='print tmap stats')
    parser.add_argument('--ground-metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='ground metric for OT calculations, only works in free support v2 and soon with Ground Metric class in all! .')
    parser.add_argument('--ground-metric-normalize', type=str, default='log', choices=['log', 'max', 'none', 'median', 'mean'],
                        help='ground metric normalization to consider! ')
    parser.add_argument('--ground-metric-eff', action='store_true', help='memory efficient calculation of ground metric')
    parser.add_argument('--dist-normalize', action='store_true', help='normalize distances by act num samples')
    parser.add_argument('--clip-gm', action='store_true', help='to clip ground metric')
    parser.add_argument('--clip-min', action='store', type=float, default=0,
                       help='Value for clip-min for gm')
    parser.add_argument('--clip-max', action='store', type=float, default=5,
                       help='Value for clip-max for gm')
    parser.add_argument('--use_batchnorm', action='store_true')
    parser.add_argument('--linear_bias', action='store_true')
    parser.add_argument('--ckpt-type', type=str, default='best', choices=['best', 'final'], help='which checkpoint to load')
    parser.add_argument('--disable_bias', action='store_true', help='disable bias in the neural network layers')
    parser.add_argument('--activation-histograms', action='store_true', help='utilize activation histograms')
    
    # GAMF

    # REPAIR

    return parser


class Trainer(Base_Trainer):
    def __init__(
        self, args,
        model=None,
        model_args=None,
        train_loader=None,
        val_loader=None,
        num_classes=None,
        optimizer=None,
        lr_scheduler=None,
        criterion=None,
        device=None
        ):
        super().__init__(args, model, model_args, train_loader, val_loader, num_classes, optimizer, lr_scheduler, criterion, device)

        merging_models_list = []
        for model_path in args.pretrained_url:
           cur_model, cur_model_args = prepare_model(args, self.logger, use_batchnorm=args.use_batchnorm, linear_bias=args.linear_bias, model_args={'mean': FB_RESNET_MEAN, 'std': FB_RESNET_STD})
           prepare_pretrained(cur_model, [model_path], self.logger)
           merging_models_list.append(cur_model)

        core_fn = prepare_merging_method(self.args, self.model_args, self.logger)

        self.model = core_fn(self.model, merging_models_list)
           