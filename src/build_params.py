import json
import itertools

params_file_name = 'params_to_select'

network_type = 'ncde'
epochs = [20]
hidden_size = [8, 12]
batch_size = [5]
lr = [0.001]


params_grid = list(itertools.product(epochs, hidden_size, batch_size, lr))
args = dict()
args['models_grid'] = {'model_%d' % step:
{    
    'epochs': epochs,
    'hidden size': hidden_size,
    'batch size': batch_size,
    'lr': lr
} 
for step, (epochs, hidden_size, batch_size, lr) in enumerate(params_grid)
}
args['network_type'] = network_type

import io

with io.open('../data/' + params_file_name + '.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(args,
                      indent = 4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(str(str_))


"""
results = dict()
results['model0'] = [1,2,3]
results['model1'] = [2,3,4]

with io.open('../data/' + 'list' + '.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(results,
                      indent = 4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(str(str_))
"""