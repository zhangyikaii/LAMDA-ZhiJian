from zhijian.models.utils import get_class_from_module

def prepare_merging_method(args, model_args, logger):
    cur_merging_class = get_class_from_module(f'zhijian.models.model_merging.method.{args.merging_mode.lower()}', args.merging_mode)
    method_list = ['Soup', 'OT', 'GAMF', 'REPAIR']
    if args.merging_mode in method_list:
        core = cur_merging_class(args, model_args, logger).core
    else:
        raise NotImplementedError

    return core
