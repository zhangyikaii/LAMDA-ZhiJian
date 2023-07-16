from zhijian.models.utils import get_command_line_parser, pprint, init_device
from zhijian.trainers.base import prepare_trainer, prepare_args

if __name__ == '__main__':
    args, parser = get_command_line_parser()

    args = prepare_args(args, parser)

    pprint(vars(args))

    init_device(args)

    trainer = prepare_trainer(args)
    trainer.fit()
    trainer.test()
