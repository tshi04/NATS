from pyrouge import Rouge155

r = Rouge155()
r.system_dir = '/home/tian/rouge/system_summaries/'
r.model_dir = '/home/tian/rouge/model_summaries/'
r.system_filename_pattern = 'sum.(\d+).txt'
r.model_filename_pattern = 'sum.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
print(output_dict)
