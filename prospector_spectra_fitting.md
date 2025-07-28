When fitting an optical spectrum using ```prospector```, we can follow the example provided in https://github.com/bd-j/exspect/blob/main/fitting/psb_params.py. 
There are several points that need to be noticed compared to broadband SED fitting:

- mask sky lines [Hanuschik et al. (2003)](https://ui.adsabs.harvard.edu/abs/2003A%26A...407.1157H/abstract)
- sedmodel.PolySpecModel() is used instead of sedmodel.SedModel().
- spectral smoothing
- continuum removal (for phot+spec fittings)
- emission line marginalization (2x faster, but gas properties will be unconstrained)
- optional: adding an outlier model

A detailed description of these settings can be found in [Johnson et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J).

Besides, there are other tricks when performing spectrophorometric SED fitting with ```prospector```:

### Q1. SFH is extremely bursty

When fitting a spectrophorometric SED,  you may get an unexpectedly SFH in the non-parametric form (e.g., fixed time bins).

*Possible solutions:* Using a prior of truncated t-distribution on the logsfr_ratios will be more robust than the original one, which allows the ratios
to reach an unphisically large value.   

Here's an example of using truncated t-distribution:
```python
nbins = 7
model_params['logsfr_ratios']['prior'] = priors.FastTruncatedEvenStudentTFreeDeg2(hw=np.full(nbins-1, 10),
                                                                                  sig=np.full(nbins-1, 0.3))
```

The source code of ```prospector``` should be slighly modified, otherwise you will get a ```NotImplementedError```.   
Modify the class in ```prospect/models/prior.py```, line 648:
```python
class FastTruncatedEvenStudentTFreeDeg2(Prior):

    prior_params = ['hw', 'sig']

    def __init__(self, hw=0.0, sig=1.0, parnames=[], name='', ):
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)

        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name

        self.hw, self.sig = hw, sig

        if np.any(self.hw <= 0.0):
            raise ValueError('hw must be greater than 0.0')

        if np.any(self.sig <= 0.0):
            raise ValueError('sig must be greater than 0.0')

        self.const1 = np.sqrt(1.0 + 0.5*(self.hw**2.0))
        self.const2 = 2.0 * self.sig * self.hw
        self.const3 = self.const2**2.0
        self.const4 = 2.0 * (self.hw**2.0)

    def __len__(self):
        return len(self.hw)

    def __call__(self, x):
        if not hasattr(x, "__len__"):
            if np.abs(x) <= self.hw:
                return np.log(self.const1 / (self.const2 * (1 + 0.5*(x / self.sig)**2.0)**1.5))
            else:
                return np.NINF
        else:
            ret = np.log(self.const1 / (self.const2 * (1 + 0.5*(x / self.sig)**2.0)**1.5))
            bad = np.abs(x) > self.hw
            ret[bad] = np.NINF
            return ret

    def scale(self):
        return self.sig

    def loc(self):
        return 0.0

    def invcdf_numerator(self, x):
        return -1.0 * (self.const3 * x**2.0 - self.const3 * x + (self.sig * self.hw)**2.0)

    def invcdf_denominator(self, x):
        return self.const4 * x**2.0 - self.const4 * x - self.sig**2.0

    def unit_transform(self, x):
        f = (((x > 0.5) & (x <= 1.0)) * np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x)) -
             ((x >= 0.0) & (x <= 0.5)) * np.sqrt(self.invcdf_numerator(x) / self.invcdf_denominator(x)))
        return f

    def sample(self):
        return self.unit_transform(np.random.rand())
        
    def bounds(self, **kwargs):
        return (-self.hw, self.hw)
```

### Q2. Redshift can not be fitted properly despite lots of emission lines

FSPS imposes a minimal line-width on the model spectra:  
https://github.com/dfm/python-fsps/issues/229  
https://github.com/cconroy20/fsps/blob/master/src/add_nebular.f90#L45  
If you are working on spatially resolved data, the observed linewidth can be much narrower than this minimal value, which can be problematic.

*Possible solutions:* We can smooth the observed spectrum before fitting, or we can edit the ```add_nebular.f90``` file and recompile FSPS.
