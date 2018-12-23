'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import sys

def show_progress(curr_, total_, time=""):
    prog_ = int(round(100.0*float(curr_)/float(total_)))
    dstr = '[' + '>'*int(round(prog_/4)) + ' '*(25-int(round(prog_/4))) + ']'
    sys.stdout.write(dstr + str(prog_) + '%' + time +'\r')
    sys.stdout.flush()
    
def str2bool(input_):
    if input_.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

