import pandas as pd
import foehnix
ellboegen = pd.read_csv('../data/ellboegen.csv', delimiter=';', skipinitialspace=True)
sattelberg = pd.read_csv('../data/sattelberg.csv', delimiter=';', skipinitialspace=True)
ellboegen.head()
data = pd.merge(ellboegen, sattelberg, on='timestamp', how='outer', suffixes=('', '_crest'), sort=True)
data.index = pd.to_datetime(data.timestamp, unit='s')
train = data.iloc[:-10].copy()
test = data.iloc[-10:].copy()
train['diff_t'] = train['t_crest'] + 10.27 - train['t']
ddfilter = {'dd': [43, 223], 'dd_crest': [90, 270]}
model = foehnix.Foehnix('diff_t', train, concomitant='ff', filter_method=ddfilter, switch=True, verbose=True)
model.summary()
model.plot(['loglik', 'loglikcontribution', 'coef'], log=True)
plt.show()