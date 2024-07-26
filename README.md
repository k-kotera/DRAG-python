## DRAG (Discord Range Aware Gathering) - python
This is an unofficial python implementation of DRAG algorithm, which efficiently searches for time series discords. DRAG was first introduced in [1], but was not named, hence the name in [2].

## Example
```python
from drag import drag

window_size = 100
discord_defining_range = 10.
C, C_dist = drag(X, window_size, discord_defining_range)
```

## References
1. Yankov, Dragomir, Eamonn Keogh, and Umaa Rebbapragada. "Disk aware discord discovery: Finding unusual time series in terabyte sized datasets." Knowledge and Information Systems 17 (2008): 241-262.
2. Nakamura, Takaaki, et al. "Merlin: Parameter-free discovery of arbitrary length anomalies in massive time series archives." 2020 IEEE international conference on data mining (ICDM). IEEE, 2020.

