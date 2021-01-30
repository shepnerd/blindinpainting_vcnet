from options.train_options import TrainOptions
from trainer.blindinpaint_trainer import BlindInpaint_Trainer

if __name__ == '__main__':
    config = TrainOptions().parse()

    print(config)

    trainer = BlindInpaint_Trainer(config)

    trainer.run()