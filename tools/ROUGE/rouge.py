from pyrouge import Rouge155

r = Rouge155()
r.system_dir = '/home/tian/newsroom2/sum_data/rouge/system_summaries/'
r.model_dir = '/home/tian/newsroom2/sum_data/rouge/model_summaries/'
r.system_filename_pattern = 'sum.(\d+).txt'
r.model_filename_pattern = 'sum.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
output_dict = r.output_to_dict(output)
print(output)
