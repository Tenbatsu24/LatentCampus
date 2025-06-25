from nnssl.training.nnsslTrainer import AbstractTrainer


class KVConsisConTrainer(AbstractTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_batch_size = 8
