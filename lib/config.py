import yaml
from pathlib import Path


class Config:
    def __init__(self, config_file: str | Path):
        self.config_file = Path(config_file)
        with open(self.config_file) as f:
            self.data = yaml.safe_load(f)

        # defines most common settings for IntelliSense
        self.experiment_name = self.experiment_name if "experiment_name" in self.data else None
        self.comment = self.comment if "comment" in self.data else None
        
        self.epochs = self.epochs if "epochs" in self.data else None
        self.batch_size = self.batch_size if "batch_size" in self.data else None
        self.device = self.device if "device" in self.data else None
        self.dataset = self.dataset if "dataset" in self.data else None
        
        self.optimizer = self.optimizer if "optimizer" in self.data else None
        self.scheduler = self.scheduler if "scheduler" in self.data else None
        
        self.train: Config = self.train if "train" in self.data else None
        self.test: Config = self.test if "test" in self.data else None
        
        self.generator: Config = self.generator if "generator" in self.data else None
        self.discriminator: Config = (
            self.discriminator if "discriminator" in self.data else None
        )
        
        self.arch = self.arch if "arch" in self.data else None
        self.lr = self.lr if "lr" in self.data else None
        self.input_dim = self.input_dim if "input_dim" in self.data else None
        self.ct_dim = self.ct_dim if "ct_dim" in self.data else None
        self.features = self.features if "features" in self.data else None


        self.input = self.input if "input" in self.data else None
        self.output = self.output if "output" in self.data else None
        self.model_name = self.model_name if "model_name" in self.data else None
        self.checkpoint = self.checkpoint if "checkpoint" in self.data else None
        
    def __str__(self) -> str:
        return f"Config({str(self.data)})"

    def __repr__(self) -> str:
        return f"Config({str(self.data)})"

    def __getattr__(self, name: str):
        try:
            value = self.data[name]
            if isinstance(value, dict):
                value = Config.from_dict(value)
                return value
            return value
        except KeyError as e:
            raise AttributeError(f"No such attribute {e} in configuration")

    def merge_cli(self, args):
        """Merge command line arguments with configuration data.

        This method takes as input an instance of argparse's Namespace object which contains
        all the parsed command-line options and their values. It iterates over these options
        and updates corresponding key-value pairs in the `data` dictionary, where keys are
        the names of command line arguments and values are their respective values from
        the command line. This allows configuration to be set via both config files and
        command line parameters.

        Args:
            args (argparse.Namespace): An instance of argparse's Namespace object which contains all the parsed command-line options and their values.

        Returns:
            None
        """
        for arg in vars(args).keys():
            self.data[arg] = getattr(args, arg)

    def keys(self):
        return self.data.keys()

    @classmethod
    def from_dict(cls, data):
        """Instantiates a new instance of this class from an input dictionary.

        Args:
            data (dict): The configuration data to be used for instantiation.

        Returns:
            An instance of the class with its 'data' attribute set to the
            provided `data` argument.
        """
        config = cls.__new__(cls)
        config.data = data
        return config
