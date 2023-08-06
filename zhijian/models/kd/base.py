def prepare_kd_loss(args):
    if args.kd_mode == 'logits':
        from zhijian.models.kd.loss.logits import Logits
        kd_criterion = Logits()
    elif args.kd_mode == 'st':
        from zhijian.models.kd.loss.st import SoftTarget
        kd_criterion = SoftTarget(args.temperature)
    elif args.kd_mode == 'at':
        from zhijian.models.kd.loss.at import AT
        kd_criterion = AT(args.p)
    elif args.kd_mode == 'fitnet':
        from zhijian.models.kd.loss.fitnet import Hint
        kd_criterion = Hint()
    elif args.kd_mode == 'nst':
        from zhijian.models.kd.loss.nst import NST
        kd_criterion = NST()
    elif args.kd_mode == 'pkt':
        from zhijian.models.kd.loss.pkt import PKTCosSim
        kd_criterion = PKTCosSim()
    elif args.kd_mode == 'rkd':
        from zhijian.models.kd.loss.rkd import RKD
        kd_criterion = RKD(args.w_dist, args.w_angle)
    elif args.kd_mode == 'sp':
        from zhijian.models.kd.loss.sp import SP
        kd_criterion = SP()
    elif args.kd_mode == 'sobolev':
        from zhijian.models.kd.loss.sobolev import Sobolev
        kd_criterion = Sobolev()
    elif args.kd_mode == 'cc':
        from zhijian.models.kd.loss.cc import CC
        kd_criterion = CC(args.gamma, args.P_order)
    elif args.kd_mode == 'lwm':
        from zhijian.models.kd.loss.lwm import LwM
        kd_criterion = LwM()
    elif args.kd_mode == 'irg':
        from zhijian.models.kd.loss.irg import IRG
        kd_criterion = IRG(args.w_irg_vert, args.w_irg_edge, args.w_irg_tran)
    elif args.kd_mode == 'refilled':
        from zhijian.models.kd.loss.refilled import Refilled_stage1, Refilled_stage2
        kd_criterion = [Refilled_stage1(args.temperature), Refilled_stage2(args.T2)]
    elif args.kd_mode == 'customized':
        from zhijian.models.kd.loss.customized import Customized
        kd_criterion = Customized()
    else:
        raise Exception('Invalid kd mode...')


    return kd_criterion
