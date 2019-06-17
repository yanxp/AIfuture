import mxnet as mx
import argparse
parser = argparse.ArgumentParser(description='Test Face Recogonition task')
parser.add_argument('--pretrained-detector', dest="pdetect",
                        help="detector checkpoint prefix", default="./models/testR50")
parser.add_argument('--detector-epoch', dest='depoch', default=4, type=int)
parser.add_argument('--save', default='models/finalR50')
parser.add_argument('--save_epoch', default=0, type=int)
args = parser.parse_args()

sym, arg_params, aux_params = mx.model.load_checkpoint('models/R50', 0)
model = mx.mod.Module(symbol=sym)
model.set_params(arg_params, aux_params)

_, arg_params, aux_params = mx.model.load_checkpoint(args.pdetect, args.depoch) # trained model,epoch
model.set_params(arg_params, aux_params, allow_missing = True)
model.save_checkpoint(args.save, args.save_epoch)