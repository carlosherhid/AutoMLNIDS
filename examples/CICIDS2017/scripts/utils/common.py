"""
Utility module containing common functions used by other scripts.
"""
import inspect
import io
import json
import logging
import os
import random
import shutil
import sys
import typing
from ctypes import Array, Structure
from ctypes import Union as CUnion
from ctypes import _Pointer, _SimpleCData
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from json import JSONEncoder

import numpy as np
import torch
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel
from autogluon.tabular.predictor import TabularPredictor

SAFE_CPU_PERCENTAGE = 0.7
DEFAULT_PLOT_STYLE = os.path.join(os.path.dirname(__file__), 'presentation.mplstyle')


def get_default_cpu_number():
    return round(SAFE_CPU_PERCENTAGE * os.cpu_count())


def load_predictor(directory: str):
    # require_version_match=False / require_py_version_match=False: the
    # cluster's AutoGluon (1.6.x on Python 3.11) is newer than what predictors
    # were originally trained with (e.g. AutoGluon 1.1.1 on Python 3.10 for the
    # CICIDS2017 predictor) - AutoGluon refuses to load across either gap by
    # default. Same-major-line upgrades (1.x -> 1.6.x) have been fine in
    # practice here; a 0.8.x predictor would be a different, riskier case.
    predictor = TabularPredictor.load(
        directory, verbosity=0, require_version_match=False, require_py_version_match=False)
    assert isinstance(predictor, TabularPredictor)
    predictor.evaluate_predictions
    patch_stale_predictor(predictor)
    return predictor


def patch_stale_predictor(predictor):
    """Predictors pickled under an older AutoGluon (e.g. 1.1.1, CICIDS2017's
    original predictor) can be missing attributes that later AutoGluon
    versions' code unconditionally reads on the (feature generator / model)
    objects restored from that pickle - unpickling restores __dict__
    directly without re-running __init__, so the restored instance simply
    lacks whatever attribute was added after the predictor was originally
    saved, even though the currently-installed class expects it.
    require_version_match=False only silences AutoGluon's own compatibility
    *warning*, it doesn't backfill missing attributes. Each sub-patch below
    was added reactively as a specific AttributeError surfaced when calling
    predict_proba()/evaluate() on CICIDS2017's original (1.1.1) predictor
    under the cluster's AutoGluon 1.6.2 - there may be more; add them here
    as they turn up rather than patching individual call sites. Call this
    right after TabularPredictor.load() and before any predict_proba()/
    evaluate() call.
    """
    _patch_stale_feature_generators(predictor)
    _patch_stale_trainer_models(predictor)


def _patch_stale_feature_generators(predictor):
    """`passthrough`, read by AbstractFeatureGenerator.transform() since
    AutoGluon added optional passthrough-feature support: default to
    disabled, matching the original predictor's actual behavior since it
    predates the attribute entirely."""
    def patch_one(generator):
        if not hasattr(generator, 'passthrough'):
            generator.passthrough = False
        nested = getattr(generator, 'generators', None)
        if not nested:
            return
        for stage in nested:
            for sub_generator in (stage if isinstance(stage, (list, tuple)) else [stage]):
                patch_one(sub_generator)

    patch_one(predictor._learner.feature_generator)


def _patch_stale_trainer_models(predictor):
    """AutoGluon model objects (TabularNeuralNetTorchModel is the only model
    type this project ever loads out of a predictor) gained several new
    instance attributes over time that AbstractModel's methods/properties
    read unconditionally - same story as _patch_stale_feature_generators():
    a predictor pickled before an attribute existed simply lacks it.
    Patched at the class level rather than by walking
    predictor._trainer.get_model_names() and load_model()-ing each one,
    which would force-load every model in the ensemble from disk just to
    patch it; a class-level default only takes effect for instances that
    don't already have their own instance attribute (old pickles), so it
    can't shadow a correctly-set value on a freshly-trained model. Known
    attributes (add more here as they surface - this has been a recurring
    one-at-a-time discovery):
      - _model_specific_feature_generators: read by
        AbstractModel._preprocess_model_specific() as `== "NOTSET"` - that
        string is the class's own sentinel default (resolved lazily on
        first real use), not a guessed value.
      - _temperature_scalar: read by the AbstractModel.temperature_scalar
        property as `is not None` - None means "no temperature scaling",
        the correct default for a predictor trained before temperature
        scaling existed.
    """
    class_level_defaults = {
        '_model_specific_feature_generators': 'NOTSET',
        '_temperature_scalar': None,
    }
    for attr, default in class_level_defaults.items():
        if not hasattr(TabularNeuralNetTorchModel, attr):
            setattr(TabularNeuralNetTorchModel, attr, default)


def load_model_from_predictor(model_dir: str):
    model_name = os.path.join(
        os.path.basename(os.path.realpath(os.path.join(model_dir, os.pardir))),
        os.path.basename(os.path.realpath(model_dir)))

    predictor_dir = os.path.join(model_dir, os.pardir, os.pardir, os.pardir)
    predictor = load_predictor(predictor_dir)
    predictor.persist_models(models=[model_name])

    model = predictor._trainer.load_model(model_name=model_name)

    assert isinstance(model, TabularNeuralNetTorchModel)
    return model, predictor, model_name


def set_seed(default=0):
    """Function for setting the seed"""
    seed = int(os.environ.get('PYTHONHASHSEED', default))

    np.random.seed(seed)
    random.seed(seed)

    try:
        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass
    return seed


def create_dir(name: str, overwrite=False):
    """Function to create a directory and, in case, overwrite or
    backup the old one if already present. If already exists,
    the old one is backed up.

    Args:
        name (str): name of the directory to be created
    """
    try:
        os.makedirs(name)
    except FileExistsError:
        if overwrite is None:
            pass
        elif overwrite is False:
            os.rename(name, name + '_backup_' + datetime.now().strftime('%m-%d-%Y_%H-%M-%S'))
            os.makedirs(name)
        elif overwrite is True:
            shutil.rmtree(name)
            os.makedirs(name)


def get_logger(name: str, filepath: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """Function to create a logger, or return the existing one"""
    if name not in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        handlers = [logging.StreamHandler()]
    else:
        logger = logging.getLogger(name)
        handlers = logger.handlers
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if filepath and not any(isinstance(x, logging.FileHandler) for x in logger.handlers):
        handlers.append(logging.FileHandler(filepath, mode='w'))
    for handle in handlers:
        handle.setLevel(log_level)
        handle.setFormatter(formatter)
        logger.addHandler(handle)
    return logger


def deep_get_size(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([deep_get_size(v, seen) for v in obj.values()])
        size += sum([deep_get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += deep_get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (io.TextIOBase, io.BufferedIOBase, io.RawIOBase,
                                                           io.IOBase, str, bytes, bytearray)):
        size += sum([deep_get_size(i, seen) for i in obj])

    return size


class CDataJSONEncoder(JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 120
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 15
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get('indent') is None:
            kwargs.update({'indent': 1})
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o, *_):
        """Encode JSON object *o* with respect to single line lists."""
        o = self.default(o)

        if isinstance(o, (list, tuple)):
            if self._put_on_single_line(o):
                return '[' + ', '.join(self.encode(el) for el in o) + ']'
            self.indentation_level += 1
            output = [self.indent_str + self.encode(el) for el in o]
            self.indentation_level -= 1
            return '[\n' + ',\n'.join(output) + '\n' + self.indent_str + ']'

        if isinstance(o, dict):
            if o:
                if self._put_on_single_line(o):
                    return '{ ' + ', '.join(f'{self.encode(k)}: {self.encode(el)}' for k, el in o.items()) + ' }'
                self.indentation_level += 1
                output = [
                    self.indent_str + f'{json.dumps(k)}: {self.encode(v)}' for k, v in o.items()]
                self.indentation_level -= 1
                return '{\n' + ',\n'.join(output) + '\n' + self.indent_str + '}'
            return '{}'

        if isinstance(o, str):  # escape newlines
            o = o.replace('\n', '\\n')
            return f'"{o}"'
        return json.dumps(o)

    def iterencode(self, o, _one_shot: bool = False):
        """Required to also work with `json.dump`."""
        return self.encode(o, _one_shot)

    def _put_on_single_line(self, obj):
        return self._primitives_only(obj) and len(obj) <= self.MAX_ITEMS and len(str(obj)) - 2 <= self.MAX_WIDTH

    def _primitives_only(self, obj: typing.Union[list, tuple, dict]):
        if isinstance(obj, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in obj)

        if isinstance(obj, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in obj.values())

        raise ValueError(f'No primitive for {obj}')

    @ property
    def indent_str(self) -> str:
        """Indent the string according to the indentation level set"""

        if isinstance(self.indent, int):
            return ' ' * (self.indentation_level * self.indent)

        if isinstance(self.indent, str):
            return self.indentation_level * self.indent

        raise ValueError(
            f'indent must either be of type int or str (is: {type(self.indent)})')

    def default(self, o):
        if inspect.isclass(o):
            return o.__name__

        if isinstance(o, (Array, list, set)):
            return [self.default(e) for e in o]

        if isinstance(o, _Pointer):
            return self.default(o.contents) if o else None

        if isinstance(o, _SimpleCData):
            return self.default(o.value)

        if isinstance(o, (bool, int, float, str)):
            return o

        if o is None:
            return o

        if isinstance(o, Enum):
            return o.value

        if isinstance(o, (Structure, CUnion)):
            result = {}
            anonymous = getattr(o, '_anonymous_', [])

            for key, *_ in getattr(o, '_fields_', []):
                value = getattr(o, key)

                # private fields don't encode
                if key.startswith('_'):
                    continue

                if key in anonymous:
                    result.update(self.default(value))
                else:
                    result[key] = self.default(value)

            return result

        if is_dataclass(o):
            if hasattr(o, 'to_json'):
                return o.to_json()
            return {k.name: self.default(getattr(o, k.name)) for k in fields(o)}

        if isinstance(o, dict):
            if o and not isinstance(next(iter(o), None), (int, float, str, bool)):
                return [{'key': self.default(k), 'value': self.default(v)} for k, v in o.items()]
            return {k: self.default(v) for k, v in o.items()}

        if isinstance(o, tuple):
            if hasattr(o, '_asdict'):
                return self.default(o._asdict())
            return [self.default(e) for e in o]

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, np.integer):
            return int(o)

        if isinstance(o, np.floating):
            return float(o)

        try:
            return JSONEncoder.default(self, o)
        except TypeError:
            try:
                return o.__dict__
            except AttributeError:
                return str(o)
