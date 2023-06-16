# miniCHAMP

The code is still underdeveploment.
Please do not distributed the code at current stage.

Documentation might not be accurate. 

To install a package that includes a setup.py file, open a command or terminal window and:

1. cd into the root directory where setup.py is located.

2. Enter: python setup.py install.


Note:
mesa datacollector

from functools import partial, reduce
def collect(self, model):
        """Collect all the data for the given model object."""
        def rgetattr(obj, attr, *args):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)
            return reduce(_getattr, [obj] + attr.split('.'))
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                # Check if Lambda operator
                if isinstance(reporter, types.LambdaType):
                    self.model_vars[var].append(reporter(model))
                # Check if model attribute
                elif isinstance(reporter, str):
                    #self.model_vars[var].append(getattr(model, reporter, None))
                    self.model_vars[var].append(rgetattr(model, reporter, None))
                # Check if function with arguments
                elif isinstance(reporter, list):
                    self.model_vars[var].append(reporter[0](*reporter[1]))
                # TODO: Check if method of a class, as of now it is assumed
                # implicitly if the other checks fail.
                else:
                    self.model_vars[var].append(reporter())
        if self.agent_reporters:
            agent_records = self._record_agents(model)
            self._agent_records[model.schedule.steps] = list(agent_records)
