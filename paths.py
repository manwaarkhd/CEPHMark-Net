class Paths(object):
    logs_root_path = "./logs/statistics"

    @staticmethod
    def backbone_weights_path(name: str = None):
        if name == "vgg16":
            return "./models/backbones/vgg16"
        elif name == "vgg19":
            return "./models/backbones/vgg19"
        else:
            raise ValueError("\'{}\' backbone doesn't exists in your paths.py file".format(name))

    @staticmethod
    def dataset_root_path(name: str = None):
        if name == "isbi":
            return "./datasets/ISBI Dataset"
        elif name == "pku":
            return "./datasets/PKU Dataset"
        else:
            raise ValueError("\'{}\' dataset doesn't exists in your paths.py file".format(name))
