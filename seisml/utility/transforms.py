import obspy


class ButterworthPassFilter:
    """
    standard seismic filtering on a Stream or Trace object

    Args:
        filter_type: (string): type of filter in [bandpass, bandstop, highpass, lowpass]
        min_freq (float): minimum corner frequency for the pass/stop (not used in lowpass)
        max_freq (float): maximum corner frequency for the pass/stop (not used in highpass)
        corners (int): filter corners/order
        zerophase (bool): If True, apply filter once forwards and once backwards. This results in twice the filter order
         but zero phase shift in the resulting filtered trace.
        source (string): the data source to filter, default: raw

    Raises:
        TransformException: if a filter in not in a supporting list

    Returns:
        data: a modified dictionary with filters applied
    """
    def __init__(self, filter_type='bandpass', min_freq=0.0, max_freq=10.0, corners=2, zerophase=True, source='raw'):
        self._filters = ['highpass', 'lowpass', 'bandpass', 'bandstop']

        if not isinstance(filter_type, str):
            raise TransformException(f'filters must be a str, got {type(filter_type)}')
        if filter_type not in self._filters:
            raise TransformException(f'must provide a supported filter {self._filters}')
        if not isinstance(min_freq, float):
            raise TransformException(f'min_freq must be a float, got {type(min_freq)}')
        if not isinstance(max_freq, float):
            raise TransformException(f'max_freq must be a float, got {type(max_freq)}')
        if not isinstance(corners, int):
            raise TransformException(f'corners must be a int, got {type(corners)}')
        if not isinstance(zerophase, bool):
            raise TransformException(f'zerophase must be a bool, got {type(zerophase)}')

        self.filter_type = filter_type
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.corners = corners
        self.zerophase = zerophase
        self.source = source

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TransformException(f'data must be of type dict, got {type(data)}')
        if self.source not in data.keys():
            raise TransformException(f'source must be a key of data, got {data.keys()}', ', '.join(data.keys()))

        transformed = data[self.source].copy()

        # if not isinstance(transformed, obspy.core.stream.Stream) xor not isinstance(transformed, obspy.core.trace.Trace):
        #     raise TransformException(f'data source must be a obspy Stream or Trace, got {type(transformed)}')

        if self.filter_type in ['bandpass', 'bandstop']:
            transformed.filter(
                self.filter_type,
                freqmin=self.min_freq,
                freqmax=self.max_freq,
                corners=self.corners,
                zerophase=self.zerophase
            )
        elif self.filter_type in ['highpass', 'lowpass']:
            transformed.filter(
                self.filter_type,
                freq=self.min_freq if self.filter_type == 'highpass' else self.max_freq,
                corners=self.corners,
                zerophase=self.zerophase
            )

        data[self.filter_type] = transformed
        return data

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'filter_type: {self.filter_type}, '
            f'min_freq: {self.min_freq}, '
            f'max_freq: {self.max_freq}, '
            f'corners: {self.corners}, '
            f'zerophase: {self.zerophase}, '
            f'source: {self.source})'
        )


class TransformException(Exception):
    """
    Exception class for errors when working with transforms in seisml.
    """
    pass