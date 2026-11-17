"""
Guard shared by the exporters (export_qat_int8.py, export_mlp_to_header.py).

The firmware applies a ReLU after every hidden Linear. A checkpoint does not record which activation the network was trained with (a Tanh
has no parameters), so an exporter that rebuilds the network from the checkpoint alone would export a tanh network as a ReLU one without any
error, and its own check would still pass (it compares the firmware arithmetic with the same ReLU rebuild). The AutoML search can choose tanh.

The exporters therefore need the architecture of the network: --architecture_json takes the automl_model_summary.json written by
describe_automl_model.py next to the predictor; the exporter stops if the summary has a layer type the firmware does not implement or if
its Linear layers are not those of the checkpoint. --assume_relu is the explicit way of saying that the network is known to be ReLU-only.
"""
import json

SUPPORTED_LAYER_TYPES = ('Linear', 'ReLU', 'Dropout', 'BatchNorm1d', 'Softmax')   # BatchNorm is folded or absent in the checkpoint; Softmax does not change the argmax


def add_arguments(parser):
    parser.add_argument('--architecture_json', default=None,
                        help='automl_model_summary.json of the AutoML model (describe_automl_model.py): the exporter checks that the network only has '
                             'layers the firmware implements (ReLU after every hidden Linear) and that its Linear layers are those of the checkpoint')
    parser.add_argument('--assume_relu', action='store_true',
                        help='export without checking the architecture, for a network known to be ReLU-only (the checkpoint does not record its activation)')


def verify(args, linear_shapes):
    """Stops the export (SystemExit) unless the network is verified, or explicitly assumed, to be ReLU-only.
    linear_shapes: (out_features, in_features) of the Linear layers of the checkpoint, in order."""
    if args.architecture_json is None:
        if not args.assume_relu:
            raise SystemExit(
                'The checkpoint does not record its activation function and the firmware only implements ReLU: a tanh network would be exported '
                'as a ReLU one without any error. Give --architecture_json <automl_model_summary.json> (describe_automl_model.py) to check it, '
                'or --assume_relu if the network is known to use ReLU.')
        print('WARNING: --assume_relu: the activation function is assumed to be ReLU, not verified')
        return
    with open(args.architecture_json, encoding='utf-8') as f:
        layers = json.load(f)['layers']
    unsupported = sorted({layer['type'] for layer in layers} - set(SUPPORTED_LAYER_TYPES))
    if unsupported:
        raise SystemExit(f'{args.architecture_json}: the network has layer types the firmware does not implement: {", ".join(unsupported)} '
                         f'(supported: {", ".join(SUPPORTED_LAYER_TYPES)}). It cannot be exported; retrain it with ReLU.')
    declared = [(layer['outputs'], layer['inputs']) for layer in layers if layer['type'] == 'Linear']
    actual = [tuple(int(v) for v in shape) for shape in linear_shapes]
    # the first layer of the AutoGluon network can take more inputs than the data has columns (a column with few distinct values becomes several
    # inputs) and the replica is built on the columns: for that layer only its width is compared; every other layer is compared completely
    same = (len(declared) == len(actual) and all(d[0] == a[0] for d, a in zip(declared, actual))
            and all(d[1] == a[1] for d, a in zip(declared[1:], actual[1:])))
    if not same:
        raise SystemExit(f'{args.architecture_json}: its Linear layers (out, in) {declared} are not those of the checkpoint {actual} '
                         f'(the input width of the first layer is not compared): wrong summary for this checkpoint')
    print(f'Architecture verified against {args.architecture_json}: ReLU only, the Linear layers match the checkpoint '
          f'({actual[0][1]} inputs in the checkpoint, {declared[0][1]} in the AutoGluon network)')
