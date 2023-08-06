def prepare_reg_loss(args):
    if args.reg_mode == 'l2sp':
        from zhijian.models.regularization.loss.l2sp import L2SP
        reg_criterion = L2SP(args.reuse_keys, args.reg_alpha, args.reg_beta)
    elif args.reg_mode == 'delta':
        from zhijian.models.regularization.loss.delta import DELTA
        reg_criterion = DELTA(args.reuse_keys, args.reg_alpha, args.reg_beta)
    elif args.reg_mode == 'bss':
        from zhijian.models.regularization.loss.bss import BSS
        reg_criterion = BSS(args.reg_alpha)
    elif args.reg_mode == 'customized':
        from zhijian.models.regularization.loss.customized import Customized
        reg_criterion = Customized(args.reuse_keys)
    else:
        raise NotImplementedError

    return reg_criterion
