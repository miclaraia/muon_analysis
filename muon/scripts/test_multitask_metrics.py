from muon.dissolving.multitask import Metrics
# import logging
# logging.getLogger(__name__).setLevel(logging.INFO)
m = Metrics()
print(m.add(100, [.1,.2,.3,.4], [.5,.6,.7,.8], [.001,.001], [.002,.002]))

m.save('/tmp/test_metrics.pkl')
