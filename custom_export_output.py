from tensorflow.python.estimator.export.export_output import ExportOutput
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY

DEFAULT_SERVING_SIGNATURE_DEF_KEY = DEFAULT_SERVING_SIGNATURE_DEF_KEY
SIGNATURE_OUTPUT_NAME = "scores"


class CustomClassificationOutput(ExportOutput):
    """Represents the output of a classification head.

  If only scores is set, it is interpreted as providing a score for every class
  in order of class ID.

  If both classes and scores are set, they are interpreted as zipped, so each
  score corresponds to the class at the same index.  Clients should not depend
  on the order of the entries.
  """

    def __init__(self, scores=None):
        """Constructor for `ClassificationOutput`.
        Args:
          scores: dict of string to `Tensor`.

        Raises:
          ValueError: if neither classes nor scores is set, or one of them is not a
              `Tensor` with the correct dtype.
        """
        if (scores is not None
                and not (isinstance(scores, dict))):
            raise ValueError('Classification scores must be a dict; '
                             'got {}'.format(scores))

        self._scores = scores

    @property
    def scores(self):
        return self._scores

    def as_signature_def(self, receiver_tensors):
        return signature_def_utils.predict_signature_def(receiver_tensors, self.scores)
