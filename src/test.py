if __name__ == '__main__':
    from settings.register_func import register_func
    from pytorch_helper.launcher import Tester
    from pytorch_helper.launcher.parse import MainArg

    Tester(MainArg, register_func).run()
