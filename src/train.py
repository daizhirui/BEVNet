if __name__ == '__main__':
    from settings.register_func import register_func
    from pytorch_helper.launcher import Trainer
    from pytorch_helper.launcher.parse import MainArg

    Trainer(MainArg, register_func).run()
