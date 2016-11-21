import tensorflow as tf

from argparse import ArgumentParser
from src.model import CryptoNet
from src.config import *


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--msg-len', type=int,
                        dest='msg_len', help='message length',
                        metavar='MSG_LEN', default=MSG_LEN)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='Number of Epochs in Adversarial Training',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    with tf.Session() as sess:
        crypto_net = CryptoNet(sess, msg_len=options.msg_len, epochs=options.epochs,
                               batch_size=options.batch_size, learning_rate=options.learning_rate)

        crypto_net.train()

if __name__ == '__main__':
    main()
